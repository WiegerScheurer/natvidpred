"""
V-JEPA 2 Latent-Space Future Prediction
========================================
Functionally equivalent to the PredNet pixel-prediction script.

Uses learned decoder to reconstruct pixels from predicted tokens.

Prediction pipeline
-------------------
1. Encode all video frames with the V-JEPA 2 encoder.

2. Predict future tokens using V-JEPA 2's *learned* predictor network:
      context tokens [0..ctx-1]  +  target position IDs  →  predicted target tokens
   Falls back to velocity extrapolation if the predictor API is unavailable.

3. Pixel reconstruction via trained convolutional decoder:
   - Maps patch tokens [n_out, P, D] → pixel frames [n_out, 3, H, W]

4. Pixel-space + latent-space metrics, comparison GIF, PCA trajectory plot.

5. Raw predicted latent tensors saved as .pt files for downstream use.
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


# ─── Decoder Architecture ──────────────────────────────────────────────────────

class VJepa2Decoder(nn.Module):
    """
    Convolutional upsampler: V-JEPA 2 patch tokens → pixel frame.
    
    Input : [B, N_SPATIAL, EMBED_DIM]   e.g. [B, 576, 1408]
    Output: [B, 3, img_size, img_size]  e.g. [B, 3, 384, 384]
    """
    def __init__(self, n_spatial=576, embed_dim=1408,
                 decoder_dim=512, img_size=384):
        super().__init__()
        self.patch_grid = int(n_spatial ** 0.5)   # 24
        self.img_size   = img_size

        # Per-patch linear projection
        self.proj = nn.Linear(embed_dim, decoder_dim)

        # Convolutional upsampling: 24 → 48 → 96 → 192 → 384
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
        """
        patch_tokens : [B, N_spatial, embed_dim]
        returns      : [B, 3, img_size, img_size]
        """
        B = patch_tokens.shape[0]
        x = self.proj(patch_tokens)                              # [B, N, D_dec]
        x = x.permute(0, 2, 1)                                   # [B, D_dec, N]
        x = x.reshape(B, -1, self.patch_grid, self.patch_grid)   # [B, D_dec, 24, 24]
        x = self.up(x)                                            # [B, 3, 384, 384]
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, (self.img_size, self.img_size),
                               mode='bilinear', align_corners=False)
        return x


# ─── Configuration ─────────────────────────────────────────────────────────────

HF_MODEL_NAME        = "facebook/vjepa2-vitg-fpc64-384"
DECODER_CHECKPOINT   = "/project/3018078.02/natvidpred_workspace/decoder_checkpoints/vjepa2_decoder_best.pt"
VIDEO_DIR            = '/project/3018078.02/MEG_ingmar/shorts/'
VIDEO_PATH           = VIDEO_DIR + "bw_testclip_bouwval.mp4"

# Analogous to PredNet's nt=15 / context_frames=10
NUM_OUTPUT_TOKENS    = 15   # total temporal token steps to process + display
CONTEXT_TOKENS       = 10   # real context tokens; rest are predicted (= 5 future)

# V-JEPA 2 model config for facebook/vjepa2-vitg-fpc64-384
TUBELET_SIZE         = 2    # raw frames per temporal token
NUM_VIDEO_FRAMES     = 64   # frames loaded per clip (fpc64)

# Experiment loops
TEMPORAL_STRIDE_VALUES  = [1, 2]
REVERSE_INPUT_OPTIONS   = [False, True]

# Fallback latent prediction (only used if predictor API unavailable)
PREDICTION_DAMPING   = 0.9

CUSTOM_SUFFIX        = ""


# ─── Metric helpers (identical to PredNet script) ──────────────────────────────

def compute_ssim(gray_a, gray_b):
    gray_a = gray_a.astype(np.float32)
    gray_b = gray_b.astype(np.float32)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a  = cv2.GaussianBlur(gray_a, (7, 7), 1.5)
    mu_b  = cv2.GaussianBlur(gray_b, (7, 7), 1.5)
    mu_a2 = mu_a * mu_a; mu_b2 = mu_b * mu_b; mu_ab = mu_a * mu_b
    s_a2  = cv2.GaussianBlur(gray_a * gray_a, (7, 7), 1.5) - mu_a2
    s_b2  = cv2.GaussianBlur(gray_b * gray_b, (7, 7), 1.5) - mu_b2
    s_ab  = cv2.GaussianBlur(gray_a * gray_b, (7, 7), 1.5) - mu_ab
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
        a_t = _t.from_numpy(g_a).float()[None, None]
        b_t = _t.from_numpy(g_b).float()[None, None]
        with _t.no_grad():
            flow = m(a_t, b_t, iters=12, test_mode=True)[-1][0].permute(1,2,0).cpu().numpy()
        return min(float(np.mean(np.sqrt(flow[...,0]**2+flow[...,1]**2)))/50.0, 1.0)
    except Exception:
        flow = cv2.calcOpticalFlowFarneback(
            g_a, g_b, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return min(float(np.mean(mag))/50.0, 1.0)


def compute_blur_loss(gt_u8, pr_u8):
    gt_v = float(cv2.Laplacian(gt_u8, cv2.CV_32F).var())
    pr_v = float(cv2.Laplacian(pr_u8, cv2.CV_32F).var())
    return float(np.clip(1.0 - pr_v / (gt_v + 1e-8), 0.0, 1.0))


def compute_phase_stats(values, boundary_idx):
    v = np.asarray(values, dtype=np.float32)
    s = int(np.clip(boundary_idx, 0, v.size))
    def ms(a): return (float('nan'), float('nan')) if a.size == 0 else (float(np.mean(a)), float(np.std(a)))
    cm, cs = ms(v[:s]); fm, fs = ms(v[s:])
    delta  = float(fm-cm) if (np.isfinite(cm) and np.isfinite(fm)) else float('nan')
    return dict(n_context=v[:s].size, n_future=v[s:].size,
                context_mean=cm, context_std=cs, future_mean=fm, future_std=fs,
                delta_future_minus_context=delta)


# ─── V-JEPA 2 predictor ────────────────────────────────────────────────────────

@torch.inference_mode()
def try_vjepa2_predictor(model, encoder_hidden_states, ctx_tok, fut_tok, n_spatial):
    """
    Call V-JEPA 2's learned predictor using the correct API signature.

    Parameters
    ----------
    encoder_hidden_states : [1, (ctx_tok + fut_tok) * n_spatial, D]  full encoded sequence
    ctx_tok : number of context temporal tokens
    fut_tok : number of future temporal tokens to predict
    n_spatial : number of spatial patches per temporal token

    Returns
    -------
    (predicted_future [fut_tok, n_spatial, D] on CPU, method_name_str)
    or (None, None) if call fails.
    """
    device = encoder_hidden_states.device
    D      = encoder_hidden_states.shape[-1]
    seq_len = encoder_hidden_states.shape[1]

    predictor = getattr(model, 'predictor', None)
    if predictor is None:
        print("  model.predictor attribute not found.")
        return None, None

    # Print the predictor's forward signature once so the user can see it
    try:
        print(f"  model.predictor.forward{inspect.signature(predictor.forward)}")
    except Exception:
        pass

    # Create integer masks: 1 = include this position, 0 = mask out
    # (predictor's gather() requires int64, not bool)
    # Context positions: [0, ctx_tok * n_spatial)
    # Target positions:  [ctx_tok * n_spatial, (ctx_tok + fut_tok) * n_spatial)
    ctx_mask = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    ctx_mask[:, :ctx_tok * n_spatial] = 1
    
    tgt_mask = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    tgt_mask[:, ctx_tok * n_spatial:(ctx_tok + fut_tok) * n_spatial] = 1

    try:
        # Call predictor with encoder_hidden_states + binary masks
        out = predictor(
            encoder_hidden_states=encoder_hidden_states,
            context_mask=[ctx_mask],        # list of 1-batch binary masks
            target_mask=[tgt_mask],         # list of 1-batch binary masks
        )
        # Extract predicted future tokens from output
        # out is BaseModelOutput with .last_hidden_state
        predicted_full = out.last_hidden_state  # [1, seq_len, D]
        predicted_future = predicted_full[:, ctx_tok*n_spatial:(ctx_tok+fut_tok)*n_spatial, :]
        predicted_future = predicted_future.reshape(fut_tok, n_spatial, D).cpu()
        print("  Predictor call succeeded ✓")
        return predicted_future, "VJEPA2Predictor(encoder_hidden_states, context_mask, target_mask)"
    except Exception as e:
        print(f"  Predictor call failed: {e}")
        print("  Falling back to velocity extrapolation.")
        return None, None


def velocity_extrapolation_fallback(context_per_tok, fut_tok, damping=0.9):
    """
    Mean-velocity damped extrapolation — fallback when predictor is unavailable.
    """
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
    return torch.stack(preds, dim=0)   # [fut_tok, P, D]


# ─── Latent distance ───────────────────────────────────────────────────────────

def latent_cosine_distance(predicted, gt):
    """Per-token cosine distance [0, 2]; 0 = identical."""
    p = F.normalize(predicted.float().mean(dim=1), dim=-1)
    g = F.normalize(gt.float().mean(dim=1),        dim=-1)
    return [float(1.0 - v) for v in (p * g).sum(dim=-1).clamp(-1, 1)]


# ─── Video I/O ─────────────────────────────────────────────────────────────────

def load_video_frames(path, num_frames, img_size, stride=1):
    cap, frames = cv2.VideoCapture(path), []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (img_size, img_size)),
                                   cv2.COLOR_BGR2RGB))
        for _ in range(stride - 1):
            if not cap.grab():
                break
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


@torch.inference_mode()
def encode_clip(model, processor, frames_rgb):
    """Returns [1, T*P, D] on CPU."""
    video  = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames_rgb])
    inputs = processor(video, return_tensors="pt")
    pv     = inputs["pixel_values_videos"].cuda()
    return model.get_vision_features(pv).cpu()


def to_temporal(tokens, num_frames, tubelet_size):
    """[1, T*P, D] or [T*P, D]  →  [T, P, D]"""
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0)
    T = num_frames // tubelet_size
    P = tokens.shape[0] // T
    return tokens.reshape(T, P, tokens.shape[-1])


# ─── PCA trajectory ────────────────────────────────────────────────────────────

def save_pca_trajectory(all_tok_full, predicted_all_tok, ctx_tok, save_path):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  sklearn not available; skipping PCA plot.")
        return

    lib_vecs  = all_tok_full.float().mean(dim=1).numpy()
    pred_vecs = predicted_all_tok.float().mean(dim=1).numpy()
    proj      = PCA(n_components=2).fit_transform(
                    np.vstack([lib_vecs, pred_vecs]))
    T_lib     = lib_vecs.shape[0]
    T_pred    = pred_vecs.shape[0]
    lib_2d    = proj[:T_lib]
    pred_2d   = proj[T_lib:]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lib_2d[:, 0], lib_2d[:, 1], 'o-', color='steelblue',
            markersize=4, linewidth=1, alpha=0.5, label='GT tokens (full video)')
    ax.scatter(*lib_2d[0],  marker='s', s=60,  color='steelblue', zorder=5)
    ax.scatter(*lib_2d[-1], marker='*', s=100, color='steelblue', zorder=5)
    ax.plot(pred_2d[:ctx_tok, 0], pred_2d[:ctx_tok, 1], 'o-', color='green',
            markersize=6, linewidth=2, label=f'Context (t=0..{ctx_tok-1})')
    if T_pred > ctx_tok:
        ax.plot(pred_2d[ctx_tok-1:, 0], pred_2d[ctx_tok-1:, 1], 'o--', color='red',
                markersize=6, linewidth=2,
                label=f'Predicted future (t={ctx_tok}..{T_pred-1})')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('V-JEPA 2 Latent Trajectory (PCA)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120)
    plt.close(fig)
    print(f'  Saved PCA trajectory → {save_path}')


# ─── Visualisation helpers ─────────────────────────────────────────────────────

def _try_font(size=10):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
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

    gl  = "Future (GT)"  if fi >= ctx_tok else "Context (GT)"
    pl_ = "Future (Pred)" if fi >= ctx_tok else "Context (Decoder)"
    draw.text((w//2-55, 2),     gl,  fill=wh, font=font)
    draw.text((w+w//2-65, 2),   pl_, fill=wh, font=font)
    canvas.paste(Image.fromarray(gt_f), (0, title_h))
    canvas.paste(Image.fromarray(pr_f), (w, title_h))

    x0 = pl; y0 = title_h+h+pt; x1 = cw-pr_; y1 = ch-pb
    nm = len(metric_series); rg = 5; rh = (y1-y0-(nm-1)*rg)//nm

    for mi, (mname, mvals, mmax, mcol) in enumerate(metric_series):
        ry0 = y0+mi*(rh+rg); ry1 = ry0+rh
        draw.line([(x0,ry0),(x0,ry1)], fill=(180,180,180), width=1)
        draw.line([(x0,ry1),(x1,ry1)], fill=(180,180,180), width=1)
        draw.text((40,ry0), mname, fill=(220,220,220), font=font)
        draw.text((x0-25,ry0), f"{mmax:.1f}", fill=(150,150,150), font=font)
        if 0 <= ctx_tok-1 < len(mvals) > 1:
            sx = int(x0+(ctx_tok-1)/(len(mvals)-1)*(x1-x0))
            draw.line([(sx,ry0),(sx,ry1)], fill=(80,80,80), width=1)
        if len(mvals) == 1:
            pts = [((x0+x1)//2, ry1)]
        else:
            pts = [(int(x0+k/(len(mvals)-1)*(x1-x0)),
                    int(ry1-(v/max(mmax,1e-9))*(ry1-ry0)))
                   for k,v in enumerate(mvals)]
        if len(pts) > 1:
            draw.line(pts, fill=mcol, width=2)
        ck = max(0, min(fi, len(mvals)-1))
        cx_, cy_ = pts[ck]; r = 3
        draw.ellipse((cx_-r,cy_-r,cx_+r,cy_+r), fill=(255,60,60))
        draw.text((cx_-40, max(ry0,cy_-15)), f"{mvals[ck]:.3f}",
                  fill=(255,120,120), font=font)

    draw.text((x0, y1+4),    "0",                              fill=(180,180,180), font=font)
    draw.text((x1-22, y1+4), str(len(metric_series[0][1])-1), fill=(180,180,180), font=font)
    return canvas


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    suffix     = f"_decoder{CUSTOM_SUFFIX}"
    video_stem = Path(VIDEO_PATH).stem
    run_ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir    = Path('predictions_vjepa2') / f'{video_stem}_{run_ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Saving outputs to:', out_dir)

    print(f"Loading V-JEPA 2: {HF_MODEL_NAME}")
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    model     = AutoModel.from_pretrained(HF_MODEL_NAME)
    model.cuda().eval()
    img_size  = processor.crop_size['height']
    print(f"  Crop size: {img_size}×{img_size}")
    
    print(f"Loading decoder: {DECODER_CHECKPOINT}")
    decoder = VJepa2Decoder(n_spatial=576, embed_dim=1408, decoder_dim=512, img_size=img_size)
    ckpt = torch.load(DECODER_CHECKPOINT, map_location='cpu')
    decoder.load_state_dict(ckpt['state_dict'])
    decoder = decoder.cuda().eval()
    print(f"  Decoder loaded successfully")
    
    font      = _try_font(10)

    with open(out_dir / 'run_config.txt', 'w') as f:
        f.write('\n'.join([
            f"encoder_model: {HF_MODEL_NAME}",
            f"decoder_checkpoint: {DECODER_CHECKPOINT}",
            f"video_path: {VIDEO_PATH}",
            f"num_output_tokens: {NUM_OUTPUT_TOKENS}",
            f"context_tokens: {CONTEXT_TOKENS}",
            f"future_tokens: {NUM_OUTPUT_TOKENS - CONTEXT_TOKENS}",
            f"tubelet_size: {TUBELET_SIZE}",
            f"num_video_frames: {NUM_VIDEO_FRAMES}",
            f"temporal_stride_values: {TEMPORAL_STRIDE_VALUES}",
            f"reverse_input_options: {REVERSE_INPUT_OPTIONS}",
            "pixel metrics: rmse, 1-ssim, blur_loss, 1-edge_f1, optical_flow_epe",
            "latent metric: cosine_distance",
            "reconstruction: learned decoder (trained on 10240 token-frame pairs)",
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
                print("Input order: REVERSED")
            else:
                print("Input order: FORWARD")

            # ── Encode ───────────────────────────────────────────────────────
            print("Encoding video clip with V-JEPA 2 …")
            raw_tokens   = encode_clip(model, processor, frames)
            all_tok_full = to_temporal(raw_tokens, NUM_VIDEO_FRAMES, TUBELET_SIZE)
            T_lib, P, D  = all_tok_full.shape
            n_out   = min(NUM_OUTPUT_TOKENS, T_lib)
            ctx_tok = min(CONTEXT_TOKENS, n_out)
            fut_tok = n_out - ctx_tok
            print(f"  Library: [T={T_lib}, P={P}, D={D}]  |  "
                  f"window: {n_out} ({ctx_tok} ctx + {fut_tok} fut)")

            # ── Predict ──────────────────────────────────────────────────────
            predictor_used = None
            if fut_tok > 0:
                # Pass the full window (context + future positions) to the predictor
                window_tokens = raw_tokens[:, :(ctx_tok + fut_tok) * P, :].cuda()
                pred_future, predictor_used = try_vjepa2_predictor(
                    model, window_tokens, ctx_tok, fut_tok, P)

                if pred_future is not None:
                    predicted_all_tok = torch.cat(
                        [all_tok_full[:ctx_tok], pred_future], dim=0)
                else:
                    print("  Using velocity extrapolation fallback.")
                    pred_future = velocity_extrapolation_fallback(
                        all_tok_full[:ctx_tok], fut_tok, PREDICTION_DAMPING)
                    predicted_all_tok = torch.cat(
                        [all_tok_full[:ctx_tok], pred_future], dim=0)
            else:
                predicted_all_tok = all_tok_full[:n_out].clone()

            # ── Save latents ─────────────────────────────────────────────────
            lat_path = out_dir / f'latents{suffix}{mode_tag}{stride_tag}.pt'
            torch.save({
                'predicted_all_tok': predicted_all_tok,    # [n_out, P, D]
                'gt_tok_window':     all_tok_full[:n_out], # [n_out, P, D]
                'gt_tok_full':       all_tok_full,          # [T_lib, P, D]
                'ctx_tok': ctx_tok, 'fut_tok': fut_tok,
                'predictor_used': predictor_used,
                'n_spatial': P, 'embed_dim': D,
                'tubelet_size': TUBELET_SIZE,
                'temporal_stride': temporal_stride,
                'reverse': reverse_input_frames,
            }, str(lat_path))
            print(f'  Saved latents → {lat_path}')

            # ── PCA plot ─────────────────────────────────────────────────────
            save_pca_trajectory(all_tok_full, predicted_all_tok, ctx_tok,
                                 out_dir / f'pca{suffix}{mode_tag}{stride_tag}.png')

            # ── Decoder-based reconstruction ───────────────────────────────────
            print("Reconstructing pixels via learned decoder …")
            with torch.inference_mode():
                decoder_device = next(decoder.parameters()).device
                pred_tokens_dev = predicted_all_tok.to(decoder_device)
                predicted_pixels = decoder(
                    pred_tokens_dev  # [n_out, P, D]
                ).cpu()  # [n_out, 3, img_size, img_size]
            
            # Convert to uint8 RGB frames
            gif_frames = [
                (predicted_pixels[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                for i in range(predicted_pixels.shape[0])
            ]
            print(f"  Generated {len(gif_frames)} decoder frames")

            gt_frames     = [frames[min(t*TUBELET_SIZE, len(frames)-1)]
                             for t in range(n_out)]
            latent_errors = latent_cosine_distance(predicted_all_tok, all_tok_full[:n_out])

            # ── Pixel metrics ────────────────────────────────────────────────
            rmse_v, ssim_e_v, blur_v, edge_e_v, flow_e_v = [], [], [], [], []
            for i, (gf, pf) in enumerate(zip(gt_frames, gif_frames)):
                gf_f = gf.astype(np.float32)/255; pf_f = pf.astype(np.float32)/255
                rmse_v.append(float(np.sqrt(np.mean((gf_f-pf_f)**2))))
                gg = cv2.cvtColor(gf, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                pg = cv2.cvtColor(pf, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                ssim_e_v.append(float(1.0 - compute_ssim(gg, pg)))
                gu8 = (gg*255).astype(np.uint8); pu8 = (pg*255).astype(np.uint8)
                blur_v.append(compute_blur_loss(gu8, pu8))
                edge_e_v.append(float(1.0 - compute_edge_f1(gu8, pu8)))
                if i < n_out - 1:
                    gn = cv2.cvtColor(gt_frames[i+1], cv2.COLOR_RGB2GRAY).astype(np.uint8)
                    pn = cv2.cvtColor(gif_frames[i+1], cv2.COLOR_RGB2GRAY).astype(np.uint8)
                    flow_e_v.append(abs(compute_optical_flow_error(gu8, gn) -
                                        compute_optical_flow_error(pu8, pn)))
                else:
                    flow_e_v.append(flow_e_v[-1] if flow_e_v else 0.0)
            flow_e_v = [0.0] + flow_e_v[:-1]

            # ── Save GIFs ────────────────────────────────────────────────────
            save_gif(gif_frames, out_dir/f'predictions{suffix}{mode_tag}{stride_tag}.gif')
            save_gif(gt_frames,  out_dir/f'groundtruth{suffix}{mode_tag}{stride_tag}.gif')

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
                f"encoder_model: {HF_MODEL_NAME}",
                f"decoder_checkpoint: {DECODER_CHECKPOINT}",
                f"predictor_used: {predictor_used}",
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
            with open(spath, 'w') as f:
                f.write('\n'.join(lines)+'\n')
            print('\n'.join(lines))

            # ── Comparison GIF ────────────────────────────────────────────────
            mplot = [
                ("RMSE",        rmse_v,       0.20, (50,  180, 255)),
                ("1-SSIM",      ssim_e_v,     1.00, (140, 230, 160)),
                ("Blur Loss",   blur_v,        1.00, (255, 170,  90)),
                ("1-EdgeF1",    edge_e_v,      1.00, (250, 200, 120)),
                ("Flow EPE",    flow_e_v,      0.20, (200, 150, 255)),
                ("Latent Dist", latent_errors, 1.00, (255, 100, 100)),
            ]
            comp = [build_comparison_frame(gf, pf, fi, ctx_tok, mplot, font)
                    for fi, (gf, pf) in enumerate(
                        zip(gt_frames[1:], gif_frames[1:]), start=1)]
            cp = out_dir/f'comparison{suffix}{mode_tag}{stride_tag}.gif'
            comp[0].save(str(cp), save_all=True, append_images=comp[1:],
                         duration=150, loop=0, optimize=False)
            print(f'SAVED comparison GIF → {cp}')

    print('\nAll runs complete.')


if __name__ == "__main__":
    main()
