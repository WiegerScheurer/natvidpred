"""
V-JEPA 2 Prediction + Trained Decoder  (v2 — diagnostic + corrected predictor)
================================================================================

Root cause of the 24×24 mosaic artifact
-----------------------------------------
The decoder maps [1, 576, D] → [1, D, 24, 24] → [1, 3, 384×384].
Each of the 24×24=576 spatial positions contributes one 16×16 tile.
A mosaic of 576 near-identical miniature images means the 576 spatial tokens
within a predicted temporal step all carry essentially the SAME information
(low spatial variance inside a temporal token).

Why does this happen for predicted future tokens but not context tokens?
- Context tokens come straight from the encoder: each of the 576 patch tokens
  encodes its LOCAL spatial region → high spatial variance → correct decoding.
- The V-JEPA 2 predictor uses full self-attention over context + target tokens.
  Without the correct SPATIAL POSITIONAL ENCODING for each target token, the
  predictor cannot distinguish "which of the 576 patches am I predicting?" and
  returns a spatially uniform/global representation for every target position.
- The decoder was trained on spatially-specific encoder tokens.  Feeding it
  globally-mixed predictor tokens breaks the spatial mapping → mosaic.

Fixes applied here
------------------
1.  diagnose_tokens() — prints spatial-variance, norm and cosine-similarity
    statistics so you can confirm the hypothesis before re-training anything.

2.  GT bypass mode (USE_GT_FUTURE_TOKENS = True) — replaces predicted future
    tokens with the actual encoder tokens for those positions.  Use this to
    confirm the decoder pipeline is end-to-end correct and that the issue is
    purely in the predictor output, not in the decoder or the plumbing.

3.  Norm-matching heuristic — optionally rescales predicted token norms to
    match the mean norm of context encoder tokens.  Sometimes sufficient to
    make the spatial structure decodable when the scale is off.

4.  try_vjepa2_predictor() now prints the predictor's forward() signature and
    the spatial variance of its output, so you can see exactly what the API
    is returning.

Long-term fix
-------------
Retrain the decoder on PREDICTOR OUTPUT tokens at target positions rather than
encoder tokens.  In encode_clip_to_token_frame_pairs(), run the predictor on
each clip (masking the latter half of temporal tokens) and store its output at
target positions as the "latent" side of the training pair.  This aligns the
decoder input distribution with what it sees at inference time.
"""

import inspect
import os
from pathlib import Path
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor


# ─── Configuration ─────────────────────────────────────────────────────────────

HF_MODEL_NAME        = "facebook/vjepa2-vitg-fpc64-384"
DECODER_CHECKPOINT   = "/project/3018078.02/natvidpred_workspace/decoder_checkpoints/vjepa2_decoder_best.pt"
VIDEO_DIR            = '/project/3018078.02/MEG_ingmar/shorts/'
VIDEO_PATH           = VIDEO_DIR + "bw_testclip_bouwval.mp4"

NUM_OUTPUT_TOKENS    = 15
CONTEXT_TOKENS       = 10

TUBELET_SIZE         = 2
NUM_VIDEO_FRAMES     = 64

TEMPORAL_STRIDE_VALUES  = [1] # [1, 2]
REVERSE_INPUT_OPTIONS   = [False] # [False, True]

PREDICTION_DAMPING   = 0.9
CUSTOM_SUFFIX        = ""

# ── Diagnostic / ablation flags ───────────────────────────────────────────────

# Set True to replace predicted future tokens with GT encoder tokens.
# This lets you confirm the decoder is correct independently of the predictor.
# Expected result: decoded future frames should look like real video frames.
USE_GT_FUTURE_TOKENS = True # was false

# Set True to rescale predicted token norms to match context token norms
# before decoding.  Quick heuristic that may partially fix scale mismatch.
NORM_MATCH_PREDICTED = True


# ─── Decoder architecture ─────────────────────────────────────────────────────

class VJepa2Decoder(nn.Module):
    def __init__(self, n_spatial=576, embed_dim=1408,
                 decoder_dim=512, img_size=384):
        super().__init__()
        self.patch_grid = int(n_spatial ** 0.5)
        self.img_size   = img_size
        self.proj = nn.Linear(embed_dim, decoder_dim)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, 256, 4, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8,  128), nn.GELU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            nn.GroupNorm(8,   64), nn.GELU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            nn.GroupNorm(4,   32), nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, patch_tokens):
        B = patch_tokens.shape[0]
        x = self.proj(patch_tokens)
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, self.patch_grid, self.patch_grid)
        x = self.up(x)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, (self.img_size, self.img_size),
                               mode='bilinear', align_corners=False)
        return x


def load_decoder(checkpoint_path, device='cuda'):
    ckpt    = torch.load(checkpoint_path, map_location='cpu')
    decoder = VJepa2Decoder(
        n_spatial   = ckpt.get('n_spatial',   576),
        embed_dim   = ckpt.get('embed_dim',  1408),
        decoder_dim = ckpt.get('decoder_dim', 512),
        img_size    = ckpt.get('img_size',    384),
    )
    state_dict = ckpt.get('state_dict', ckpt)
    decoder.load_state_dict(state_dict)
    decoder.to(device).eval()
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder loaded  ({n_params/1e6:.1f}M params)")
    return decoder


# ─── Diagnostic helper ────────────────────────────────────────────────────────

def diagnose_tokens(label, tokens):
    """
    Print spatial-variance, norm and inter-step cosine-similarity statistics
    for a [T, P, D] token tensor.

    Key quantities
    --------------
    spatial_var : variance of the P=576 patch vectors WITHIN each temporal step.
                  High → tokens are spatially diverse (encoder-like).
                  Low  → tokens are spatially uniform (globally mixed = mosaic risk).
    norm        : L2 norm of each patch vector.
    inter_step_cos : cosine similarity between consecutive temporal steps
                     (pool over patches first).  Low → temporal diversity.
    """
    tokens = tokens.float()            # [T, P, D]
    T, P, D = tokens.shape

    # Spatial variance within each temporal step
    spatial_var = tokens.var(dim=1).mean().item()   # scalar

    # Per-patch norm
    norms = tokens.norm(dim=-1)                      # [T, P]
    norm_mean = norms.mean().item()
    norm_std  = norms.std().item()

    # Inter-step cosine similarity (pooled)
    pooled = F.normalize(tokens.mean(dim=1), dim=-1) # [T, D]
    if T > 1:
        cos_steps = (pooled[:-1] * pooled[1:]).sum(dim=-1)  # [T-1]
        cos_mean  = cos_steps.mean().item()
        cos_min   = cos_steps.min().item()
    else:
        cos_mean = cos_min = float('nan')

    print(f"  [{label}]  T={T} P={P} D={D}")
    print(f"    spatial_var (within step):  {spatial_var:.4f}   "
          f"← should be HIGH for encoder tokens, LOW for globally-mixed tokens")
    print(f"    patch norm:  mean={norm_mean:.3f}  std={norm_std:.3f}")
    print(f"    inter-step cosine:  mean={cos_mean:.4f}  min={cos_min:.4f}   "
          f"← near 1.0 → all future steps identical")


# ─── Norm-matching heuristic ──────────────────────────────────────────────────

def match_norms(predicted_future, context_tokens):
    """
    Rescale predicted_future so its per-patch norms match the mean norm of
    context encoder tokens.  Does NOT change spatial structure, only scale.
    This is a heuristic: it helps when the issue is purely a scale mismatch
    between the predictor and the encoder output space.
    """
    ctx_mean_norm  = context_tokens.float().norm(dim=-1).mean().item()
    pred_norms     = predicted_future.float().norm(dim=-1, keepdim=True)  # [T, P, 1]
    pred_mean_norm = pred_norms.mean().item()
    if pred_mean_norm < 1e-6:
        print("  Norm-match: predicted norms near zero, skipping.")
        return predicted_future
    scale = ctx_mean_norm / pred_mean_norm
    print(f"  Norm-match: ctx_norm={ctx_mean_norm:.3f}  "
          f"pred_norm={pred_mean_norm:.3f}  scale={scale:.3f}")
    return predicted_future * scale


# ─── Decode tokens to pixels ─────────────────────────────────────────────────

@torch.inference_mode()
def decode_tokens(predicted_per_tok, decoder, device='cuda'):
    """[T, P, D] → list of T uint8 HxWx3 numpy arrays."""
    decoded = []
    for t in range(predicted_per_tok.shape[0]):
        tok = predicted_per_tok[t].unsqueeze(0).to(device)  # [1, P, D]
        img = decoder(tok)                                   # [1, 3, H, W]
        img_np = (img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        decoded.append((img_np * 255).clip(0, 255).astype(np.uint8))
    return decoded


# ─── V-JEPA 2 predictor ───────────────────────────────────────────────────────

@torch.inference_mode()
def try_vjepa2_predictor(model, encoder_hidden_states, ctx_tok, fut_tok, n_spatial):
    """
    Call V-JEPA 2's predictor and return predicted future tokens.

    Prints the predictor's forward() signature and the spatial variance of its
    output so you can see whether the API is returning useful representations.
    """
    device  = encoder_hidden_states.device
    D       = encoder_hidden_states.shape[-1]
    seq_len = encoder_hidden_states.shape[1]

    predictor = getattr(model, 'predictor', None)
    if predictor is None:
        print("  model.predictor not found.")
        return None, None

    try:
        sig = inspect.signature(predictor.forward)
        print(f"  predictor.forward{sig}")
    except Exception:
        pass

    ctx_mask = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    ctx_mask[:, :ctx_tok * n_spatial] = 1
    tgt_mask = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    tgt_mask[:, ctx_tok * n_spatial:(ctx_tok + fut_tok) * n_spatial] = 1

    try:
        out = predictor(
            encoder_hidden_states=encoder_hidden_states,
            context_mask=[ctx_mask],
            target_mask=[tgt_mask],
        )
        predicted_full   = out.last_hidden_state          # [1, seq_len, D]
        predicted_future = predicted_full[
            :, ctx_tok * n_spatial:(ctx_tok + fut_tok) * n_spatial, :
        ]                                                  # [1, fut_tok*P, D]
        predicted_future = predicted_future.reshape(fut_tok, n_spatial, D).cpu()

        # ── Diagnostic: spatial variance of predictor output ────────────────
        spatial_var = predicted_future.float().var(dim=1).mean().item()
        print(f"  Predictor call succeeded ✓")
        print(f"  Predictor output spatial_var = {spatial_var:.6f}  "
              f"← if near 0, tokens are globally mixed → mosaic")

        method = "VJEPA2Predictor(encoder_hidden_states, context_mask, target_mask)"
        return predicted_future, method

    except Exception as e:
        print(f"  Predictor call failed: {e}")
        print("  Falling back to velocity extrapolation.")
        return None, None


def velocity_extrapolation_fallback(context_per_tok, fut_tok, damping=0.9):
    T = context_per_tok.shape[0]
    if T >= 2:
        deltas   = context_per_tok[1:] - context_per_tok[:-1]
        velocity = deltas.mean(dim=0)
        step_mag = deltas.norm(dim=-1).mean()
        vel_mag  = velocity.norm(dim=-1, keepdim=True).mean()
        if vel_mag > step_mag:
            velocity = velocity * (step_mag / vel_mag)
    else:
        velocity = torch.zeros_like(context_per_tok[-1])
    preds, last = [], context_per_tok[-1].clone()
    for step in range(fut_tok):
        nxt = last + velocity * (damping ** (step + 1))
        preds.append(nxt); last = nxt.clone()
    return torch.stack(preds, dim=0)


# ─── Latent distance ─────────────────────────────────────────────────────────

def latent_cosine_distance(predicted, gt):
    p = F.normalize(predicted.float().mean(dim=1), dim=-1)
    g = F.normalize(gt.float().mean(dim=1),        dim=-1)
    return [float(1.0 - v) for v in (p * g).sum(dim=-1).clamp(-1, 1)]


# ─── Metric helpers ───────────────────────────────────────────────────────────

def compute_ssim(gray_a, gray_b):
    gray_a = gray_a.astype(np.float32)
    gray_b = gray_b.astype(np.float32)
    c1, c2 = 0.01**2, 0.03**2
    mu_a  = cv2.GaussianBlur(gray_a, (7,7), 1.5)
    mu_b  = cv2.GaussianBlur(gray_b, (7,7), 1.5)
    mu_a2 = mu_a*mu_a; mu_b2 = mu_b*mu_b; mu_ab = mu_a*mu_b
    s_a2  = cv2.GaussianBlur(gray_a*gray_a, (7,7), 1.5) - mu_a2
    s_b2  = cv2.GaussianBlur(gray_b*gray_b, (7,7), 1.5) - mu_b2
    s_ab  = cv2.GaussianBlur(gray_a*gray_b, (7,7), 1.5) - mu_ab
    return float(np.mean(
        ((2*mu_ab+c1)*(2*s_ab+c2)) / ((mu_a2+mu_b2+c1)*(s_a2+s_b2+c2)+1e-8)))

def compute_edge_f1(g_a, g_b):
    ea = cv2.Canny(g_a, 100, 200) > 0
    eb = cv2.Canny(g_b, 100, 200) > 0
    tp = np.logical_and(ea, eb).sum()
    fp = np.logical_and(~ea, eb).sum()
    fn = np.logical_and(ea, ~eb).sum()
    d  = 2*tp + fp + fn
    return 1.0 if d == 0 else float(2*tp / d)

def compute_optical_flow_error(g_a, g_b):
    try:
        import torch as _t
        from raft_core.raft import RAFT as _RAFT
        m = _RAFT().to('cpu').eval()
        a_t = _t.from_numpy(g_a).float()[None,None]
        b_t = _t.from_numpy(g_b).float()[None,None]
        with _t.no_grad():
            flow = m(a_t, b_t, iters=12, test_mode=True)[-1][0].permute(1,2,0).cpu().numpy()
        return min(float(np.mean(np.sqrt(flow[...,0]**2+flow[...,1]**2)))/50., 1.)
    except Exception:
        flow = cv2.calcOpticalFlowFarneback(
            g_a, g_b, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        return min(float(np.mean(mag))/50., 1.)

def compute_blur_loss(gt_u8, pr_u8):
    gt_v = float(cv2.Laplacian(gt_u8, cv2.CV_32F).var())
    pr_v = float(cv2.Laplacian(pr_u8, cv2.CV_32F).var())
    return float(np.clip(1. - pr_v/(gt_v+1e-8), 0., 1.))

def compute_phase_stats(values, boundary_idx):
    v = np.asarray(values, dtype=np.float32)
    s = int(np.clip(boundary_idx, 0, v.size))
    def ms(a): return (float('nan'), float('nan')) if a.size==0 else (float(np.mean(a)), float(np.std(a)))
    cm, cs = ms(v[:s]); fm, fs = ms(v[s:])
    delta = float(fm-cm) if (np.isfinite(cm) and np.isfinite(fm)) else float('nan')
    return dict(n_context=v[:s].size, n_future=v[s:].size,
                context_mean=cm, context_std=cs, future_mean=fm, future_std=fs,
                delta_future_minus_context=delta)


# ─── Video I/O ────────────────────────────────────────────────────────────────

def load_video_frames(path, num_frames, img_size, stride=1):
    cap, frames = cv2.VideoCapture(path), []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (img_size, img_size)),
                                   cv2.COLOR_BGR2RGB))
        for _ in range(stride-1):
            if not cap.grab():
                break
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]

@torch.inference_mode()
def encode_clip(model, processor, frames_rgb):
    video  = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in frames_rgb])
    inputs = processor(video, return_tensors="pt")
    pv     = inputs["pixel_values_videos"].cuda()
    return model.get_vision_features(pv).cpu()

def to_temporal(tokens, num_frames, tubelet_size):
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0)
    T = num_frames // tubelet_size
    P = tokens.shape[0] // T
    return tokens.reshape(T, P, tokens.shape[-1])


# ─── PCA trajectory ───────────────────────────────────────────────────────────

def save_pca_trajectory(all_tok_full, predicted_all_tok, ctx_tok, save_path):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  sklearn not available; skipping PCA plot.")
        return
    lib_vecs  = all_tok_full.float().mean(dim=1).numpy()
    pred_vecs = predicted_all_tok.float().mean(dim=1).numpy()
    proj      = PCA(n_components=2).fit_transform(np.vstack([lib_vecs, pred_vecs]))
    T_lib = lib_vecs.shape[0]; T_pred = pred_vecs.shape[0]
    lib_2d = proj[:T_lib]; pred_2d = proj[T_lib:]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(lib_2d[:,0], lib_2d[:,1], 'o-', color='steelblue',
            markersize=4, linewidth=1, alpha=0.5, label='GT tokens (full video)')
    ax.scatter(*lib_2d[0],  marker='s', s=60,  color='steelblue', zorder=5)
    ax.scatter(*lib_2d[-1], marker='*', s=100, color='steelblue', zorder=5)
    ax.plot(pred_2d[:ctx_tok,0], pred_2d[:ctx_tok,1], 'o-', color='green',
            markersize=6, linewidth=2, label=f'Context (t=0..{ctx_tok-1})')
    if T_pred > ctx_tok:
        ax.plot(pred_2d[ctx_tok-1:,0], pred_2d[ctx_tok-1:,1], 'o--', color='red',
                markersize=6, linewidth=2,
                label=f'Predicted future (t={ctx_tok}..{T_pred-1})')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('V-JEPA 2 Latent Trajectory (PCA)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(str(save_path), dpi=120); plt.close(fig)
    print(f'  Saved PCA trajectory → {save_path}')


# ─── Visualisation helpers ────────────────────────────────────────────────────

def _try_font(size=10):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()

def save_gif(frames, path, ms=150, skip_first=True):
    pil = [Image.fromarray(f) for f in (frames[1:] if skip_first else frames)]
    if pil:
        pil[0].save(str(path), save_all=True, append_images=pil[1:],
                    duration=ms, loop=0, optimize=False)

def build_comparison_frame(gt_f, pr_f, fi, ctx_tok, metric_series, font,
                            title_h=20, plot_h=350, pad=(36,12,12,22)):
    h, w = gt_f.shape[:2]
    pl, pr_, pt, pb = pad
    cw, ch = w*2, title_h+h+plot_h
    canvas = Image.new('RGB', (cw, ch), (0,0,0))
    draw   = ImageDraw.Draw(canvas)
    wh     = (255,255,255)
    gl  = "Future (GT-dec)"  if fi >= ctx_tok else "Context (GT-dec)"
    pl_ = "Future (Pred-dec)" if fi >= ctx_tok else "Context (Pred-dec)"
    draw.text((w//2-55,  2), gl,  fill=wh, font=font)
    draw.text((w+w//2-65,2), pl_, fill=wh, font=font)
    canvas.paste(Image.fromarray(gt_f), (0,   title_h))
    canvas.paste(Image.fromarray(pr_f), (w,   title_h))
    x0=pl; y0=title_h+h+pt; x1=cw-pr_; y1=ch-pb
    nm=len(metric_series); rg=5; rh=(y1-y0-(nm-1)*rg)//nm
    for mi,(mname,mvals,mmax,mcol) in enumerate(metric_series):
        ry0=y0+mi*(rh+rg); ry1=ry0+rh
        draw.line([(x0,ry0),(x0,ry1)], fill=(180,180,180), width=1)
        draw.line([(x0,ry1),(x1,ry1)], fill=(180,180,180), width=1)
        draw.text((40,ry0), mname, fill=(220,220,220), font=font)
        draw.text((x0-25,ry0), f"{mmax:.1f}", fill=(150,150,150), font=font)
        if 0<=ctx_tok-1<len(mvals)>1:
            sx=int(x0+(ctx_tok-1)/(len(mvals)-1)*(x1-x0))
            draw.line([(sx,ry0),(sx,ry1)], fill=(80,80,80), width=1)
        pts=([((x0+x1)//2,ry1)] if len(mvals)==1 else
             [(int(x0+k/(len(mvals)-1)*(x1-x0)),
               int(ry1-(v/max(mmax,1e-9))*(ry1-ry0)))
              for k,v in enumerate(mvals)])
        if len(pts)>1: draw.line(pts, fill=mcol, width=2)
        ck=max(0,min(fi,len(mvals)-1)); cx_,cy_=pts[ck]; r=3
        draw.ellipse((cx_-r,cy_-r,cx_+r,cy_+r), fill=(255,60,60))
        draw.text((cx_-40,max(ry0,cy_-15)), f"{mvals[ck]:.3f}",
                  fill=(255,120,120), font=font)
    draw.text((x0,y1+4),    "0",                              fill=(180,180,180), font=font)
    draw.text((x1-22,y1+4), str(len(metric_series[0][1])-1), fill=(180,180,180), font=font)
    return canvas


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    gt_tag     = "_GT" if USE_GT_FUTURE_TOKENS else ""
    nm_tag     = "_NM" if (NORM_MATCH_PREDICTED and not USE_GT_FUTURE_TOKENS) else ""
    suffix     = f"_decoder{gt_tag}{nm_tag}{CUSTOM_SUFFIX}"
    video_stem = Path(VIDEO_PATH).stem
    run_ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir    = Path('predictions_vjepa2') / f'{video_stem}_{run_ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Saving outputs to:', out_dir)

    if USE_GT_FUTURE_TOKENS:
        print("\n*** GT BYPASS MODE: future tokens taken from encoder, not predictor ***\n")

    print(f"Loading V-JEPA 2 encoder: {HF_MODEL_NAME}")
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    model     = AutoModel.from_pretrained(HF_MODEL_NAME)
    model.cuda().eval()
    img_size  = processor.crop_size['height']
    print(f"  Crop size: {img_size}×{img_size}")
    font      = _try_font(10)

    print(f"\nLoading pixel decoder: {DECODER_CHECKPOINT}")
    decoder = load_decoder(DECODER_CHECKPOINT, device='cuda')

    with open(out_dir / 'run_config.txt', 'w') as f:
        f.write('\n'.join([
            f"model: {HF_MODEL_NAME}",
            f"decoder_checkpoint: {DECODER_CHECKPOINT}",
            f"video_path: {VIDEO_PATH}",
            f"use_gt_future_tokens: {USE_GT_FUTURE_TOKENS}",
            f"norm_match_predicted: {NORM_MATCH_PREDICTED}",
            f"num_output_tokens: {NUM_OUTPUT_TOKENS}",
            f"context_tokens: {CONTEXT_TOKENS}",
            f"future_tokens: {NUM_OUTPUT_TOKENS - CONTEXT_TOKENS}",
            f"tubelet_size: {TUBELET_SIZE}",
            f"num_video_frames: {NUM_VIDEO_FRAMES}",
            f"temporal_stride_values: {TEMPORAL_STRIDE_VALUES}",
            f"reverse_input_options: {REVERSE_INPUT_OPTIONS}",
            "pixel_reconstruction: trained_decoder",
        ]) + '\n')

    for temporal_stride in TEMPORAL_STRIDE_VALUES:
        for reverse_input_frames in REVERSE_INPUT_OPTIONS:

            mode_tag   = "_rev" if reverse_input_frames else "_fwd"
            stride_tag = f"_s{temporal_stride}"
            print(f"\n{'='*64}")
            print(f"temporal_stride={temporal_stride}  |  reverse={reverse_input_frames}")
            print(f"{'='*64}")

            frames = load_video_frames(VIDEO_PATH, NUM_VIDEO_FRAMES,
                                        img_size, temporal_stride)
            if reverse_input_frames:
                frames = list(reversed(frames))

            # ── Encode ─────────────────────────────────────────────────────
            print("Encoding video clip …")
            raw_tokens   = encode_clip(model, processor, frames)
            all_tok_full = to_temporal(raw_tokens, NUM_VIDEO_FRAMES, TUBELET_SIZE)
            T_lib, P, D  = all_tok_full.shape
            n_out   = min(NUM_OUTPUT_TOKENS, T_lib)
            ctx_tok = min(CONTEXT_TOKENS, n_out)
            fut_tok = n_out - ctx_tok
            print(f"  Library: [T={T_lib}, P={P}, D={D}]  |  "
                  f"window: {n_out} ({ctx_tok} ctx + {fut_tok} fut)")

            # Diagnostic: encoder token statistics
            diagnose_tokens("encoder GT context", all_tok_full[:ctx_tok])
            diagnose_tokens("encoder GT future",  all_tok_full[ctx_tok:n_out])

            # ── Predict / bypass ────────────────────────────────────────────
            predictor_used = None
            if USE_GT_FUTURE_TOKENS:
                # Ablation: use real encoder tokens for future positions.
                # Lets you verify the decoder pipeline without the predictor.
                pred_future    = all_tok_full[ctx_tok:n_out].clone()
                predictor_used = "GT_BYPASS"
                print("  GT bypass: using encoder tokens for future positions.")
            elif fut_tok > 0:
                window_tokens = raw_tokens[:, :(ctx_tok + fut_tok) * P, :].cuda()
                pred_future, predictor_used = try_vjepa2_predictor(
                    model, window_tokens, ctx_tok, fut_tok, P)

                if pred_future is None:
                    print("  Using velocity extrapolation fallback.")
                    pred_future = velocity_extrapolation_fallback(
                        all_tok_full[:ctx_tok], fut_tok, PREDICTION_DAMPING)
                    predictor_used = "velocity_extrapolation"

                # Diagnostic: predicted token statistics
                diagnose_tokens(f"predicted future ({predictor_used})", pred_future)

                # Optional norm-matching heuristic
                if NORM_MATCH_PREDICTED:
                    pred_future = match_norms(pred_future, all_tok_full[:ctx_tok])
                    diagnose_tokens("predicted future (after norm-match)", pred_future)
            else:
                pred_future    = torch.empty(0, P, D)
                predictor_used = "none"

            predicted_all_tok = torch.cat([all_tok_full[:ctx_tok], pred_future], dim=0)

            # ── Save latents ────────────────────────────────────────────────
            lat_path = out_dir / f'latents{suffix}{mode_tag}{stride_tag}.pt'
            torch.save({
                'predicted_all_tok': predicted_all_tok,
                'gt_tok_window':     all_tok_full[:n_out],
                'gt_tok_full':       all_tok_full,
                'ctx_tok': ctx_tok, 'fut_tok': fut_tok,
                'predictor_used': predictor_used,
                'n_spatial': P, 'embed_dim': D,
                'tubelet_size': TUBELET_SIZE,
                'temporal_stride': temporal_stride,
                'reverse': reverse_input_frames,
            }, str(lat_path))
            print(f'  Saved latents → {lat_path}')

            # ── PCA ─────────────────────────────────────────────────────────
            save_pca_trajectory(all_tok_full, predicted_all_tok, ctx_tok,
                                 out_dir / f'pca{suffix}{mode_tag}{stride_tag}.png')

            # ── Decode ──────────────────────────────────────────────────────
            print("Decoding predicted tokens …")
            decoded_pred = decode_tokens(predicted_all_tok, decoder)

            print("Decoding GT tokens …")
            decoded_gt   = decode_tokens(all_tok_full[:n_out], decoder)

            gt_frames_raw = [frames[min(t*TUBELET_SIZE, len(frames)-1)]
                             for t in range(n_out)]
            latent_errors = latent_cosine_distance(predicted_all_tok,
                                                   all_tok_full[:n_out])

            # ── Pixel metrics ────────────────────────────────────────────────
            rmse_v, ssim_e_v, blur_v, edge_e_v, flow_e_v = [], [], [], [], []
            for i, (gf, pf) in enumerate(zip(decoded_gt, decoded_pred)):
                gf_f = gf.astype(np.float32)/255; pf_f = pf.astype(np.float32)/255
                rmse_v.append(float(np.sqrt(np.mean((gf_f-pf_f)**2))))
                gg  = cv2.cvtColor(gf, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                pg  = cv2.cvtColor(pf, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                ssim_e_v.append(float(1.0-compute_ssim(gg, pg)))
                gu8=(gg*255).astype(np.uint8); pu8=(pg*255).astype(np.uint8)
                blur_v.append(compute_blur_loss(gu8, pu8))
                edge_e_v.append(float(1.0-compute_edge_f1(gu8, pu8)))
                if i < n_out-1:
                    gn=cv2.cvtColor(decoded_gt[i+1],   cv2.COLOR_RGB2GRAY).astype(np.uint8)
                    pn=cv2.cvtColor(decoded_pred[i+1], cv2.COLOR_RGB2GRAY).astype(np.uint8)
                    flow_e_v.append(abs(compute_optical_flow_error(gu8,gn)-
                                        compute_optical_flow_error(pu8,pn)))
                else:
                    flow_e_v.append(flow_e_v[-1] if flow_e_v else 0.)
            flow_e_v = [0.]+flow_e_v[:-1]

            # ── Save GIFs ────────────────────────────────────────────────────
            save_gif(decoded_pred,
                     out_dir/f'predictions{suffix}{mode_tag}{stride_tag}.gif')
            save_gif(decoded_gt,
                     out_dir/f'groundtruth_decoded{suffix}{mode_tag}{stride_tag}.gif')
            save_gif(gt_frames_raw,
                     out_dir/f'groundtruth_raw{suffix}{mode_tag}{stride_tag}.gif')

            # ── Metric summary ────────────────────────────────────────────────
            metric_series_summary = [
                ('RMSE',        rmse_v),
                ('1-SSIM',      ssim_e_v),
                ('Blur Loss',   blur_v),
                ('1-EdgeF1',    edge_e_v),
                ('Flow EPE',    flow_e_v),
                ('Latent Dist', latent_errors),
            ]
            lines = [
                f"model: {HF_MODEL_NAME}",
                f"decoder_checkpoint: {DECODER_CHECKPOINT}",
                f"predictor_used: {predictor_used}",
                f"use_gt_future_tokens: {USE_GT_FUTURE_TOKENS}",
                f"norm_match_predicted: {NORM_MATCH_PREDICTED}",
                f"input_order: {'reversed' if reverse_input_frames else 'forward'}",
                f"temporal_stride: {temporal_stride}",
                f"boundary_index (token): {ctx_tok}", '',
                'metric          ctx_mean  ctx_std  fut_mean  fut_std  delta(fut-ctx)  n_ctx  n_fut',
                '-'*86,
            ]
            for mname, mv in metric_series_summary:
                s = compute_phase_stats(mv, ctx_tok)
                lines.append(
                    f"{mname:15s} {s['context_mean']:8.4f} {s['context_std']:8.4f} "
                    f"{s['future_mean']:8.4f} {s['future_std']:8.4f} "
                    f"{s['delta_future_minus_context']:14.4f} "
                    f"{s['n_context']:6d} {s['n_future']:6d}")
            spath = out_dir/f'metrics_summary{suffix}{mode_tag}{stride_tag}.txt'
            with open(spath,'w') as f:
                f.write('\n'.join(lines)+'\n')
            print('\n'.join(lines))

            # ── Comparison GIF ───────────────────────────────────────────────
            mplot = [
                ("RMSE",        rmse_v,       0.20, (50,180,255)),
                ("1-SSIM",      ssim_e_v,     1.00, (140,230,160)),
                ("Blur Loss",   blur_v,        1.00, (255,170,90)),
                ("1-EdgeF1",    edge_e_v,      1.00, (250,200,120)),
                ("Flow EPE",    flow_e_v,      0.20, (200,150,255)),
                ("Latent Dist", latent_errors, 1.00, (255,100,100)),
            ]
            comp = [build_comparison_frame(gf, pf, fi, ctx_tok, mplot, font)
                    for fi,(gf,pf) in enumerate(
                        zip(decoded_gt[1:], decoded_pred[1:]), start=1)]
            cp = out_dir/f'comparison{suffix}{mode_tag}{stride_tag}.gif'
            comp[0].save(str(cp), save_all=True, append_images=comp[1:],
                         duration=150, loop=0, optimize=False)
            print(f'SAVED comparison GIF → {cp}')

    print('\nAll runs complete.')


if __name__ == "__main__":
    main()













# """
# V-JEPA 2 Latent-Space Future Prediction  (decoder version)
# ============================================================
# Identical prediction pipeline to the NN-retrieval version, but pixel
# reconstruction is done with the *trained* VJepa2Decoder rather than
# nearest-neighbour token lookup.

# Changes from the NN version
# ----------------------------
# - VJepa2Decoder class included and loaded from DECODER_CHECKPOINT.
# - decode_tokens()  replaces  nn_reconstruction().
# - nn_reconstruction() and latent_cosine_distance helpers are retained
#   only for the latent-distance metric; pixel reconstruction now comes
#   entirely from the decoder.
# - Decoder is kept frozen (eval + no_grad) throughout.
# """

# import inspect
# import os
# from pathlib import Path
# from datetime import datetime

# import cv2
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel, AutoVideoProcessor


# # ─── Configuration ─────────────────────────────────────────────────────────────

# HF_MODEL_NAME        = "facebook/vjepa2-vitg-fpc64-384"
# # DECODER_CHECKPOINT   = "/project/3018078.02/natvidpred_workspace/decoder_checkpoints/vjepa2_decoder_best.pt"
# DECODER_CHECKPOINT   = "/project/3018078.02/natvidpred_workspace/decoder_checkpoints/vjepa2_decoder_last.pt"
# VIDEO_DIR            = '/project/3018078.02/MEG_ingmar/shorts/'
# # VIDEO_PATH           = VIDEO_DIR + "bw_testclip_bouwval.mp4"
# # VIDEO_PATH           = VIDEO_DIR + "kanon.mp4"
# VIDEO_PATH           = VIDEO_DIR + "trein_portret.mp4"

# NUM_OUTPUT_TOKENS    = 15
# CONTEXT_TOKENS       = 10

# TUBELET_SIZE         = 2
# NUM_VIDEO_FRAMES     = 64

# TEMPORAL_STRIDE_VALUES  = [1] #[1, 2]
# REVERSE_INPUT_OPTIONS   = [False] # [False, True]

# PREDICTION_DAMPING   = 0.9
# CUSTOM_SUFFIX        = ""


# # ─── Decoder architecture (must match training script exactly) ─────────────────

# class VJepa2Decoder(nn.Module):
#     """
#     Convolutional upsampler: V-JEPA 2 patch tokens → pixel frame.

#     Input : [B, N_SPATIAL, EMBED_DIM]   e.g. [B, 576, 1408]
#     Output: [B, 3, img_size, img_size]  e.g. [B, 3, 384, 384]
#     """
#     def __init__(self, n_spatial=576, embed_dim=1408,
#                  decoder_dim=512, img_size=384):
#         super().__init__()
#         self.patch_grid = int(n_spatial ** 0.5)   # 24
#         self.img_size   = img_size

#         self.proj = nn.Linear(embed_dim, decoder_dim)

#         # Convolutional upsampling: 24 → 48 → 96 → 192 → 384
#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(decoder_dim, 256, 4, stride=2, padding=1),
#             nn.GroupNorm(16, 256), nn.GELU(),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.GroupNorm(8,  128), nn.GELU(),
#             nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
#             nn.GroupNorm(8,   64), nn.GELU(),
#             nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
#             nn.GroupNorm(4,   32), nn.GELU(),
#             nn.Conv2d(32, 3, kernel_size=3, padding=1),
#             nn.Sigmoid(),
#         )

#     def forward(self, patch_tokens):
#         """
#         patch_tokens : [B, N_spatial, embed_dim]
#         returns      : [B, 3, img_size, img_size]
#         """
#         B = patch_tokens.shape[0]
#         x = self.proj(patch_tokens)
#         x = x.permute(0, 2, 1)
#         x = x.reshape(B, -1, self.patch_grid, self.patch_grid)
#         x = self.up(x)
#         if x.shape[-1] != self.img_size:
#             x = F.interpolate(x, (self.img_size, self.img_size),
#                                mode='bilinear', align_corners=False)
#         return x


# # ─── Decoder loader ────────────────────────────────────────────────────────────

# def load_decoder(checkpoint_path, device='cuda'):
#     """
#     Load a VJepa2Decoder from a checkpoint saved by the training script.
#     All hyperparameters are read from the checkpoint so there is no need
#     to hard-code them here.

#     Returns the decoder in eval mode on `device`.
#     """
#     ckpt = torch.load(checkpoint_path, map_location='cpu')

#     decoder = VJepa2Decoder(
#         n_spatial   = ckpt.get('n_spatial',   576),
#         embed_dim   = ckpt.get('embed_dim',  1408),
#         decoder_dim = ckpt.get('decoder_dim', 512),
#         img_size    = ckpt.get('img_size',    384),
#     )
#     # Checkpoints may be 'best' (weights only) or 'last' (full training state).
#     state_dict = ckpt.get('state_dict', ckpt)
#     decoder.load_state_dict(state_dict)
#     decoder.to(device).eval()

#     n_params = sum(p.numel() for p in decoder.parameters())
#     print(f"  Decoder loaded  ({n_params/1e6:.1f}M params)  "
#           f"n_spatial={ckpt.get('n_spatial',576)}  "
#           f"embed_dim={ckpt.get('embed_dim',1408)}  "
#           f"img_size={ckpt.get('img_size',384)}")
#     return decoder


# # ─── Decoder-based pixel reconstruction ───────────────────────────────────────

# @torch.inference_mode()
# def decode_tokens(predicted_per_tok, decoder, device='cuda'):
#     """
#     Run every temporal token through the trained decoder.

#     Parameters
#     ----------
#     predicted_per_tok : [T, P, D]  (CPU tensor)
#     decoder           : VJepa2Decoder  (on `device`, eval mode)

#     Returns
#     -------
#     decoded_frames : list of T  uint8 HxWx3 numpy arrays
#     """
#     T = predicted_per_tok.shape[0]
#     decoded_frames = []

#     for t in range(T):
#         tok = predicted_per_tok[t].unsqueeze(0).to(device)  # [1, P, D]
#         img = decoder(tok)                                   # [1, 3, H, W]  in [0,1]
#         img_np = (img.squeeze(0)                             # [3, H, W]
#                      .permute(1, 2, 0)                       # [H, W, 3]
#                      .cpu().numpy())
#         decoded_frames.append((img_np * 255).clip(0, 255).astype(np.uint8))

#     return decoded_frames


# # ─── Metric helpers ────────────────────────────────────────────────────────────

# def compute_ssim(gray_a, gray_b):
#     gray_a = gray_a.astype(np.float32)
#     gray_b = gray_b.astype(np.float32)
#     c1, c2 = 0.01 ** 2, 0.03 ** 2
#     mu_a  = cv2.GaussianBlur(gray_a, (7, 7), 1.5)
#     mu_b  = cv2.GaussianBlur(gray_b, (7, 7), 1.5)
#     mu_a2 = mu_a * mu_a; mu_b2 = mu_b * mu_b; mu_ab = mu_a * mu_b
#     s_a2  = cv2.GaussianBlur(gray_a * gray_a, (7, 7), 1.5) - mu_a2
#     s_b2  = cv2.GaussianBlur(gray_b * gray_b, (7, 7), 1.5) - mu_b2
#     s_ab  = cv2.GaussianBlur(gray_a * gray_b, (7, 7), 1.5) - mu_ab
#     return float(np.mean(
#         ((2*mu_ab+c1)*(2*s_ab+c2)) / ((mu_a2+mu_b2+c1)*(s_a2+s_b2+c2)+1e-8)))


# def compute_edge_f1(g_a, g_b):
#     ea = cv2.Canny(g_a, 100, 200) > 0
#     eb = cv2.Canny(g_b, 100, 200) > 0
#     tp = np.logical_and(ea, eb).sum()
#     fp = np.logical_and(~ea, eb).sum()
#     fn = np.logical_and(ea, ~eb).sum()
#     d  = 2*tp + fp + fn
#     return 1.0 if d == 0 else float(2*tp / d)


# def compute_optical_flow_error(g_a, g_b):
#     try:
#         import torch as _t
#         from raft_core.raft import RAFT as _RAFT
#         m = _RAFT().to('cpu').eval()
#         a_t = _t.from_numpy(g_a).float()[None, None]
#         b_t = _t.from_numpy(g_b).float()[None, None]
#         with _t.no_grad():
#             flow = m(a_t, b_t, iters=12, test_mode=True)[-1][0].permute(1,2,0).cpu().numpy()
#         return min(float(np.mean(np.sqrt(flow[...,0]**2+flow[...,1]**2)))/50.0, 1.0)
#     except Exception:
#         flow = cv2.calcOpticalFlowFarneback(
#             g_a, g_b, None, pyr_scale=0.5, levels=3, winsize=15,
#             iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
#         mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         return min(float(np.mean(mag))/50.0, 1.0)


# def compute_blur_loss(gt_u8, pr_u8):
#     gt_v = float(cv2.Laplacian(gt_u8, cv2.CV_32F).var())
#     pr_v = float(cv2.Laplacian(pr_u8, cv2.CV_32F).var())
#     return float(np.clip(1.0 - pr_v / (gt_v + 1e-8), 0.0, 1.0))


# def compute_phase_stats(values, boundary_idx):
#     v = np.asarray(values, dtype=np.float32)
#     s = int(np.clip(boundary_idx, 0, v.size))
#     def ms(a): return (float('nan'), float('nan')) if a.size == 0 else (float(np.mean(a)), float(np.std(a)))
#     cm, cs = ms(v[:s]); fm, fs = ms(v[s:])
#     delta  = float(fm-cm) if (np.isfinite(cm) and np.isfinite(fm)) else float('nan')
#     return dict(n_context=v[:s].size, n_future=v[s:].size,
#                 context_mean=cm, context_std=cs, future_mean=fm, future_std=fs,
#                 delta_future_minus_context=delta)


# # ─── V-JEPA 2 predictor ────────────────────────────────────────────────────────

# @torch.inference_mode()
# def try_vjepa2_predictor(model, encoder_hidden_states, ctx_tok, fut_tok, n_spatial):
#     device  = encoder_hidden_states.device
#     D       = encoder_hidden_states.shape[-1]
#     seq_len = encoder_hidden_states.shape[1]

#     predictor = getattr(model, 'predictor', None)
#     if predictor is None:
#         print("  model.predictor attribute not found.")
#         return None, None

#     try:
#         print(f"  model.predictor.forward{inspect.signature(predictor.forward)}")
#     except Exception:
#         pass

#     ctx_mask = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
#     ctx_mask[:, :ctx_tok * n_spatial] = 1

#     tgt_mask = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
#     tgt_mask[:, ctx_tok * n_spatial:(ctx_tok + fut_tok) * n_spatial] = 1

#     try:
#         out = predictor(
#             encoder_hidden_states=encoder_hidden_states,
#             context_mask=[ctx_mask],
#             target_mask=[tgt_mask],
#         )
#         predicted_full   = out.last_hidden_state
#         predicted_future = predicted_full[:, ctx_tok*n_spatial:(ctx_tok+fut_tok)*n_spatial, :]
#         predicted_future = predicted_future.reshape(fut_tok, n_spatial, D).cpu()
#         print("  Predictor call succeeded ✓")
#         return predicted_future, "VJEPA2Predictor(encoder_hidden_states, context_mask, target_mask)"
#     except Exception as e:
#         print(f"  Predictor call failed: {e}")
#         print("  Falling back to velocity extrapolation.")
#         return None, None


# def velocity_extrapolation_fallback(context_per_tok, fut_tok, damping=0.9):
#     T = context_per_tok.shape[0]
#     if T >= 2:
#         deltas   = context_per_tok[1:] - context_per_tok[:-1]
#         velocity = deltas.mean(dim=0)
#         step_mag = deltas.norm(dim=-1).mean()
#         vel_mag  = velocity.norm(dim=-1, keepdim=True).mean()
#         if vel_mag > step_mag:
#             velocity = velocity * (step_mag / vel_mag)
#     else:
#         velocity = torch.zeros_like(context_per_tok[-1])

#     preds, last = [], context_per_tok[-1].clone()
#     for step in range(fut_tok):
#         nxt = last + velocity * (damping ** (step + 1))
#         preds.append(nxt); last = nxt.clone()
#     return torch.stack(preds, dim=0)


# # ─── Latent distance ───────────────────────────────────────────────────────────

# def latent_cosine_distance(predicted, gt):
#     """Per-token cosine distance [0, 2]; 0 = identical."""
#     p = F.normalize(predicted.float().mean(dim=1), dim=-1)
#     g = F.normalize(gt.float().mean(dim=1),        dim=-1)
#     return [float(1.0 - v) for v in (p * g).sum(dim=-1).clamp(-1, 1)]


# # ─── Video I/O ─────────────────────────────────────────────────────────────────

# def load_video_frames(path, num_frames, img_size, stride=1):
#     cap, frames = cv2.VideoCapture(path), []
#     while len(frames) < num_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(cv2.cvtColor(cv2.resize(frame, (img_size, img_size)),
#                                    cv2.COLOR_BGR2RGB))
#         for _ in range(stride - 1):
#             if not cap.grab():
#                 break
#     cap.release()
#     while len(frames) < num_frames:
#         frames.append(frames[-1].copy())
#     return frames[:num_frames]


# @torch.inference_mode()
# def encode_clip(model, processor, frames_rgb):
#     video  = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames_rgb])
#     inputs = processor(video, return_tensors="pt")
#     pv     = inputs["pixel_values_videos"].cuda()
#     return model.get_vision_features(pv).cpu()


# def to_temporal(tokens, num_frames, tubelet_size):
#     if tokens.dim() == 3:
#         tokens = tokens.squeeze(0)
#     T = num_frames // tubelet_size
#     P = tokens.shape[0] // T
#     return tokens.reshape(T, P, tokens.shape[-1])


# # ─── PCA trajectory ────────────────────────────────────────────────────────────

# def save_pca_trajectory(all_tok_full, predicted_all_tok, ctx_tok, save_path):
#     try:
#         from sklearn.decomposition import PCA
#     except ImportError:
#         print("  sklearn not available; skipping PCA plot.")
#         return

#     lib_vecs  = all_tok_full.float().mean(dim=1).numpy()
#     pred_vecs = predicted_all_tok.float().mean(dim=1).numpy()
#     proj      = PCA(n_components=2).fit_transform(
#                     np.vstack([lib_vecs, pred_vecs]))
#     T_lib  = lib_vecs.shape[0]
#     T_pred = pred_vecs.shape[0]
#     lib_2d  = proj[:T_lib]
#     pred_2d = proj[T_lib:]

#     fig, ax = plt.subplots(figsize=(7, 5))
#     ax.plot(lib_2d[:, 0], lib_2d[:, 1], 'o-', color='steelblue',
#             markersize=4, linewidth=1, alpha=0.5, label='GT tokens (full video)')
#     ax.scatter(*lib_2d[0],  marker='s', s=60,  color='steelblue', zorder=5)
#     ax.scatter(*lib_2d[-1], marker='*', s=100, color='steelblue', zorder=5)
#     ax.plot(pred_2d[:ctx_tok, 0], pred_2d[:ctx_tok, 1], 'o-', color='green',
#             markersize=6, linewidth=2, label=f'Context (t=0..{ctx_tok-1})')
#     if T_pred > ctx_tok:
#         ax.plot(pred_2d[ctx_tok-1:, 0], pred_2d[ctx_tok-1:, 1], 'o--', color='red',
#                 markersize=6, linewidth=2,
#                 label=f'Predicted future (t={ctx_tok}..{T_pred-1})')
#     ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
#     ax.set_title('V-JEPA 2 Latent Trajectory (PCA)')
#     ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(str(save_path), dpi=120)
#     plt.close(fig)
#     print(f'  Saved PCA trajectory → {save_path}')


# # ─── Visualisation helpers ─────────────────────────────────────────────────────

# def _try_font(size=10):
#     for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
#               "/System/Library/Fonts/Helvetica.ttc"):
#         try:
#             return ImageFont.truetype(p, size)
#         except Exception:
#             pass
#     return ImageFont.load_default()


# def save_gif(frames, path, ms=150, skip_first=True):
#     pil = [Image.fromarray(f) for f in (frames[1:] if skip_first else frames)]
#     if pil:
#         pil[0].save(str(path), save_all=True, append_images=pil[1:],
#                     duration=ms, loop=0, optimize=False)


# def build_comparison_frame(gt_f, pr_f, fi, ctx_tok, metric_series, font,
#                             title_h=20, plot_h=350, pad=(36,12,12,22)):
#     h, w = gt_f.shape[:2]
#     pl, pr_, pt, pb = pad
#     cw, ch = w*2, title_h+h+plot_h
#     canvas = Image.new('RGB', (cw, ch), (0,0,0))
#     draw   = ImageDraw.Draw(canvas)
#     wh     = (255,255,255)

#     gl  = "Future (GT)"   if fi >= ctx_tok else "Context (GT)"
#     pl_ = "Future (Dec)"  if fi >= ctx_tok else "Context (Dec)"
#     draw.text((w//2-55, 2),     gl,  fill=wh, font=font)
#     draw.text((w+w//2-65, 2),   pl_, fill=wh, font=font)
#     canvas.paste(Image.fromarray(gt_f), (0, title_h))
#     canvas.paste(Image.fromarray(pr_f), (w, title_h))

#     x0 = pl; y0 = title_h+h+pt; x1 = cw-pr_; y1 = ch-pb
#     nm = len(metric_series); rg = 5; rh = (y1-y0-(nm-1)*rg)//nm

#     for mi, (mname, mvals, mmax, mcol) in enumerate(metric_series):
#         ry0 = y0+mi*(rh+rg); ry1 = ry0+rh
#         draw.line([(x0,ry0),(x0,ry1)], fill=(180,180,180), width=1)
#         draw.line([(x0,ry1),(x1,ry1)], fill=(180,180,180), width=1)
#         draw.text((40,ry0), mname, fill=(220,220,220), font=font)
#         draw.text((x0-25,ry0), f"{mmax:.1f}", fill=(150,150,150), font=font)
#         if 0 <= ctx_tok-1 < len(mvals) > 1:
#             sx = int(x0+(ctx_tok-1)/(len(mvals)-1)*(x1-x0))
#             draw.line([(sx,ry0),(sx,ry1)], fill=(80,80,80), width=1)
#         if len(mvals) == 1:
#             pts = [((x0+x1)//2, ry1)]
#         else:
#             pts = [(int(x0+k/(len(mvals)-1)*(x1-x0)),
#                     int(ry1-(v/max(mmax,1e-9))*(ry1-ry0)))
#                    for k,v in enumerate(mvals)]
#         if len(pts) > 1:
#             draw.line(pts, fill=mcol, width=2)
#         ck = max(0, min(fi, len(mvals)-1))
#         cx_, cy_ = pts[ck]; r = 3
#         draw.ellipse((cx_-r,cy_-r,cx_+r,cy_+r), fill=(255,60,60))
#         draw.text((cx_-40, max(ry0,cy_-15)), f"{mvals[ck]:.3f}",
#                   fill=(255,120,120), font=font)

#     draw.text((x0, y1+4),    "0",                              fill=(180,180,180), font=font)
#     draw.text((x1-22, y1+4), str(len(metric_series[0][1])-1), fill=(180,180,180), font=font)
#     return canvas


# # ─── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     suffix     = f"_decoder{CUSTOM_SUFFIX}"
#     video_stem = Path(VIDEO_PATH).stem
#     run_ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
#     out_dir    = Path('predictions_vjepa2') / f'{video_stem}_{run_ts}'
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print('Saving outputs to:', out_dir)

#     # ── Load encoder ─────────────────────────────────────────────────────────
#     print(f"Loading V-JEPA 2 encoder: {HF_MODEL_NAME}")
#     processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
#     model     = AutoModel.from_pretrained(HF_MODEL_NAME)
#     model.cuda().eval()
#     img_size  = processor.crop_size['height']
#     print(f"  Crop size: {img_size}×{img_size}")
#     font      = _try_font(10)

#     # ── Load decoder ─────────────────────────────────────────────────────────
#     print(f"\nLoading pixel decoder: {DECODER_CHECKPOINT}")
#     decoder = load_decoder(DECODER_CHECKPOINT, device='cuda')

#     with open(out_dir / 'run_config.txt', 'w') as f:
#         f.write('\n'.join([
#             f"model: {HF_MODEL_NAME}",
#             f"decoder_checkpoint: {DECODER_CHECKPOINT}",
#             f"video_path: {VIDEO_PATH}",
#             f"num_output_tokens: {NUM_OUTPUT_TOKENS}",
#             f"context_tokens: {CONTEXT_TOKENS}",
#             f"future_tokens: {NUM_OUTPUT_TOKENS - CONTEXT_TOKENS}",
#             f"tubelet_size: {TUBELET_SIZE}",
#             f"num_video_frames: {NUM_VIDEO_FRAMES}",
#             f"temporal_stride_values: {TEMPORAL_STRIDE_VALUES}",
#             f"reverse_input_options: {REVERSE_INPUT_OPTIONS}",
#             "pixel_reconstruction: trained_decoder",
#             "pixel metrics: rmse, 1-ssim, blur_loss, 1-edge_f1, optical_flow_epe",
#             "latent metric: cosine_distance",
#         ]) + '\n')

#     for temporal_stride in TEMPORAL_STRIDE_VALUES:
#         for reverse_input_frames in REVERSE_INPUT_OPTIONS:

#             mode_tag   = "_rev" if reverse_input_frames else "_fwd"
#             stride_tag = f"_s{temporal_stride}"
#             print(f"\n{'='*64}")
#             print(f"temporal_stride={temporal_stride}  |  reverse={reverse_input_frames}")
#             print(f"{'='*64}")

#             frames = load_video_frames(VIDEO_PATH, NUM_VIDEO_FRAMES,
#                                         img_size, temporal_stride)
#             if reverse_input_frames:
#                 frames = list(reversed(frames))
#                 print("Input order: REVERSED")
#             else:
#                 print("Input order: FORWARD")

#             # ── Encode ───────────────────────────────────────────────────────
#             print("Encoding video clip with V-JEPA 2 …")
#             raw_tokens   = encode_clip(model, processor, frames)
#             all_tok_full = to_temporal(raw_tokens, NUM_VIDEO_FRAMES, TUBELET_SIZE)
#             T_lib, P, D  = all_tok_full.shape
#             n_out   = min(NUM_OUTPUT_TOKENS, T_lib)
#             ctx_tok = min(CONTEXT_TOKENS, n_out)
#             fut_tok = n_out - ctx_tok
#             print(f"  Library: [T={T_lib}, P={P}, D={D}]  |  "
#                   f"window: {n_out} ({ctx_tok} ctx + {fut_tok} fut)")

#             # ── Predict ──────────────────────────────────────────────────────
#             predictor_used = None
#             if fut_tok > 0:
#                 window_tokens = raw_tokens[:, :(ctx_tok + fut_tok) * P, :].cuda()
#                 pred_future, predictor_used = try_vjepa2_predictor(
#                     model, window_tokens, ctx_tok, fut_tok, P)

#                 if pred_future is not None:
#                     predicted_all_tok = torch.cat(
#                         [all_tok_full[:ctx_tok], pred_future], dim=0)
#                 else:
#                     print("  Using velocity extrapolation fallback.")
#                     pred_future = velocity_extrapolation_fallback(
#                         all_tok_full[:ctx_tok], fut_tok, PREDICTION_DAMPING)
#                     predicted_all_tok = torch.cat(
#                         [all_tok_full[:ctx_tok], pred_future], dim=0)
#             else:
#                 predicted_all_tok = all_tok_full[:n_out].clone()

#             # ── Save latents ─────────────────────────────────────────────────
#             lat_path = out_dir / f'latents{suffix}{mode_tag}{stride_tag}.pt'
#             torch.save({
#                 'predicted_all_tok': predicted_all_tok,
#                 'gt_tok_window':     all_tok_full[:n_out],
#                 'gt_tok_full':       all_tok_full,
#                 'ctx_tok': ctx_tok, 'fut_tok': fut_tok,
#                 'predictor_used': predictor_used,
#                 'n_spatial': P, 'embed_dim': D,
#                 'tubelet_size': TUBELET_SIZE,
#                 'temporal_stride': temporal_stride,
#                 'reverse': reverse_input_frames,
#             }, str(lat_path))
#             print(f'  Saved latents → {lat_path}')

#             # ── PCA plot ─────────────────────────────────────────────────────
#             save_pca_trajectory(all_tok_full, predicted_all_tok, ctx_tok,
#                                  out_dir / f'pca{suffix}{mode_tag}{stride_tag}.png')

#             # ── Decode to pixels ──────────────────────────────────────────────
#             # Predicted tokens  →  pixel frames via trained decoder.
#             # GT frames use the same decoder (so both sides share the same
#             # rendering pipeline, making pixel metrics meaningful).
#             print("Decoding predicted tokens to pixels …")
#             decoded_pred   = decode_tokens(predicted_all_tok, decoder, device='cuda')

#             print("Decoding GT tokens to pixels …")
#             decoded_gt     = decode_tokens(all_tok_full[:n_out], decoder, device='cuda')

#             # Raw video frames are also kept for reference GIFs
#             gt_frames_raw  = [frames[min(t * TUBELET_SIZE, len(frames) - 1)]
#                                for t in range(n_out)]

#             # ── Latent distance ───────────────────────────────────────────────
#             latent_errors  = latent_cosine_distance(predicted_all_tok,
#                                                     all_tok_full[:n_out])

#             # ── Pixel metrics (decoder pred vs decoder GT) ────────────────────
#             rmse_v, ssim_e_v, blur_v, edge_e_v, flow_e_v = [], [], [], [], []
#             for i, (gf, pf) in enumerate(zip(decoded_gt, decoded_pred)):
#                 gf_f = gf.astype(np.float32) / 255
#                 pf_f = pf.astype(np.float32) / 255
#                 rmse_v.append(float(np.sqrt(np.mean((gf_f - pf_f) ** 2))))

#                 gg  = cv2.cvtColor(gf, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
#                 pg  = cv2.cvtColor(pf, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
#                 ssim_e_v.append(float(1.0 - compute_ssim(gg, pg)))

#                 gu8 = (gg * 255).astype(np.uint8)
#                 pu8 = (pg * 255).astype(np.uint8)
#                 blur_v.append(compute_blur_loss(gu8, pu8))
#                 edge_e_v.append(float(1.0 - compute_edge_f1(gu8, pu8)))

#                 if i < n_out - 1:
#                     gn = cv2.cvtColor(decoded_gt[i+1],   cv2.COLOR_RGB2GRAY).astype(np.uint8)
#                     pn = cv2.cvtColor(decoded_pred[i+1], cv2.COLOR_RGB2GRAY).astype(np.uint8)
#                     flow_e_v.append(abs(compute_optical_flow_error(gu8, gn) -
#                                         compute_optical_flow_error(pu8, pn)))
#                 else:
#                     flow_e_v.append(flow_e_v[-1] if flow_e_v else 0.0)
#             flow_e_v = [0.0] + flow_e_v[:-1]

#             # ── Save GIFs ─────────────────────────────────────────────────────
#             # Decoded predicted frames
#             save_gif(decoded_pred,
#                      out_dir / f'predictions{suffix}{mode_tag}{stride_tag}.gif')
#             # Decoded GT frames  (decoder reconstruction of actual video)
#             save_gif(decoded_gt,
#                      out_dir / f'groundtruth_decoded{suffix}{mode_tag}{stride_tag}.gif')
#             # Raw video frames  (useful sanity reference)
#             save_gif(gt_frames_raw,
#                      out_dir / f'groundtruth_raw{suffix}{mode_tag}{stride_tag}.gif')

#             # ── Metric summary ────────────────────────────────────────────────
#             metric_series_summary = [
#                 ('RMSE',        rmse_v),
#                 ('1-SSIM',      ssim_e_v),
#                 ('Blur Loss',   blur_v),
#                 ('1-EdgeF1',    edge_e_v),
#                 ('Flow EPE',    flow_e_v),
#                 ('Latent Dist', latent_errors),
#             ]
#             lines = [
#                 f"model: {HF_MODEL_NAME}",
#                 f"decoder_checkpoint: {DECODER_CHECKPOINT}",
#                 f"predictor_used: {predictor_used}",
#                 f"input_order: {'reversed' if reverse_input_frames else 'forward'}",
#                 f"temporal_stride: {temporal_stride}",
#                 f"boundary_index (token): {ctx_tok}", '',
#                 'metric          ctx_mean  ctx_std  fut_mean  fut_std  delta(fut-ctx)  n_ctx  n_fut',
#                 '-'*86,
#             ]
#             for mname, mv in metric_series_summary:
#                 s = compute_phase_stats(mv, ctx_tok)
#                 lines.append(
#                     f"{mname:15s} {s['context_mean']:8.4f} {s['context_std']:8.4f} "
#                     f"{s['future_mean']:8.4f} {s['future_std']:8.4f} "
#                     f"{s['delta_future_minus_context']:14.4f} "
#                     f"{s['n_context']:6d} {s['n_future']:6d}")
#             spath = out_dir / f'metrics_summary{suffix}{mode_tag}{stride_tag}.txt'
#             with open(spath, 'w') as f:
#                 f.write('\n'.join(lines) + '\n')
#             print('\n'.join(lines))

#             # ── Comparison GIF ─────────────────────────────────────────────────
#             mplot = [
#                 ("RMSE",        rmse_v,       0.20, (50,  180, 255)),
#                 ("1-SSIM",      ssim_e_v,     1.00, (140, 230, 160)),
#                 ("Blur Loss",   blur_v,        1.00, (255, 170,  90)),
#                 ("1-EdgeF1",    edge_e_v,      1.00, (250, 200, 120)),
#                 ("Flow EPE",    flow_e_v,      0.20, (200, 150, 255)),
#                 ("Latent Dist", latent_errors, 1.00, (255, 100, 100)),
#             ]
#             comp = [build_comparison_frame(gf, pf, fi, ctx_tok, mplot, font)
#                     for fi, (gf, pf) in enumerate(
#                         zip(decoded_gt[1:], decoded_pred[1:]), start=1)]
#             cp = out_dir / f'comparison{suffix}{mode_tag}{stride_tag}.gif'
#             comp[0].save(str(cp), save_all=True, append_images=comp[1:],
#                          duration=150, loop=0, optimize=False)
#             print(f'SAVED comparison GIF → {cp}')

#     print('\nAll runs complete.')


# if __name__ == "__main__":
#     main()











