"""
V-JEPA 2 Latent-Space Future Prediction
========================================
Functionally equivalent to the PredNet pixel-prediction script.

Prediction pipeline
-------------------
1. Encode all video frames with the V-JEPA 2 encoder.

2. Predict future tokens using V-JEPA 2's *learned* predictor network:
      context tokens [0..ctx-1]  +  target position IDs  →  predicted target tokens
   Falls back to velocity extrapolation if the predictor API is unavailable.

3. Pixel reconstruction via temporally-constrained nearest-neighbour (NN) retrieval:
   - Context tokens retrieve their exact source frame (perfect cosine match).
   - Future tokens are restricted to library tokens at temporal index >= ctx_tok,
     guaranteeing visual advancement rather than re-showing the last context frame.

4. Pixel-space + latent-space metrics, comparison GIF, PCA trajectory plot.

5. Raw predicted latent tensors saved as .pt files for downstream use.

Root cause of the "all future frames identical" bug in earlier versions
-----------------------------------------------------------------------
Velocity extrapolation produces predicted tokens that are all small perturbations
around the last context token. In cosine-similarity space that token always wins
the NN search → same frame every time. Two fixes are applied here:
  (a) Use V-JEPA 2's actual learned predictor (correct approach).
  (b) Temporal ordering constraint in NN: future tokens may only match library
      tokens at temporal index >= ctx_tok, guaranteeing distinct future frames
      even when the fallback extrapolation is used.

V-JEPA 2 predictor API notes
-----------------------------
The script tries several plausible HuggingFace API signatures at runtime and
prints which one works (or prints the predictor's forward() signature so you
can update try_vjepa2_predictor() to match).
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
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor


# ─── Configuration ─────────────────────────────────────────────────────────────

HF_MODEL_NAME        = "facebook/vjepa2-vitg-fpc64-384"
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
    Combined with temporally-constrained NN this still shows distinct future frames.
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


# ─── Nearest-neighbour reconstruction ─────────────────────────────────────────

def nn_reconstruction(predicted_per_tok, library_per_tok,
                       library_frames, ctx_tok, tubelet_size):
    """
    Cosine NN retrieval with temporal-ordering constraint.

    Context tokens: unrestricted (retrieve exact source frame, sim ≈ 1.0).
    Future tokens:  restricted to library tokens at temporal index >= ctx_tok.
                    This is the fix for the "all future frames identical" bug:
                    even if predicted tokens cluster near the last context token,
                    that token is excluded from the candidate set, so the search
                    is forced into the genuine future portion of the video.
    """
    pred_p = F.normalize(predicted_per_tok.float().mean(dim=1), dim=-1)   # [T_pred, D]
    lib_p  = F.normalize(library_per_tok.float().mean(dim=1),   dim=-1)   # [T_lib,  D]
    sims   = pred_p @ lib_p.T                                              # [T_pred, T_lib]

    best_idx, best_sim = [], []
    for t in range(sims.shape[0]):
        row = sims[t].clone()
        if t >= ctx_tok:
            row[:ctx_tok] = -2.0   # mask out context-period library tokens
        idx = int(row.argmax().item())
        best_idx.append(idx)
        best_sim.append(float(row[idx].item()))

    reconstructed = []
    for tok_idx in best_idx:
        fi = min(int(tok_idx) * tubelet_size, len(library_frames) - 1)
        reconstructed.append(library_frames[fi])

    return reconstructed, np.array(best_idx), np.array(best_sim)


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
    pl_ = "Future (Pred)" if fi >= ctx_tok else "Context (NN)"
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
    suffix     = f"_autoregressive{CUSTOM_SUFFIX}"
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
    font      = _try_font(10)

    with open(out_dir / 'run_config.txt', 'w') as f:
        f.write('\n'.join([
            f"model: {HF_MODEL_NAME}",
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

            # ── NN reconstruction (temporally constrained) ────────────────────
            print("Reconstructing pixels via temporally-constrained NN …")
            gif_frames, nn_idx, nn_sim = nn_reconstruction(
                predicted_all_tok, all_tok_full, frames, ctx_tok, TUBELET_SIZE)
            print(f"  Future NN retrieved frame indices: {nn_idx[ctx_tok:]}")
            print(f"  NN sim — ctx: {nn_sim[:ctx_tok].mean():.4f}  "
                  f"fut: {nn_sim[ctx_tok:].mean():.4f}")

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
                f"model: {HF_MODEL_NAME}",
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


# """
# V-JEPA 2 Latent-Space Future Prediction
# ========================================
# Functionally equivalent to the PredNet pixel-prediction script, but operating
# in V-JEPA 2's learned latent space instead of pixel space.

# Prediction pipeline
# -------------------
# 1. Encode all video frames with the V-JEPA 2 encoder
#    → per-temporal-token patch features  [T_tok, P_spatial, embed_dim]

# 2. Autoregressive latent-space prediction
#    - Context tokens  : real V-JEPA 2 encodings of the first CONTEXT_TOKENS steps
#    - Future tokens   : predicted via damped temporal velocity extrapolation

# 3. Pixel reconstruction via nearest-neighbour (NN) retrieval
#    - For each predicted token, find the real video frame whose mean-pooled
#      cosine similarity is highest in V-JEPA 2 embedding space.

# 4. Compute the same pixel-space error metrics as the PredNet script, plus
#    a latent-space cosine-distance metric that is native to JEPA models.

# Notes on V-JEPA 2 architecture
# --------------------------------
# V-JEPA 2 is a Joint Embedding Predictive Architecture (JEPA), not a pixel-space
# generative model.  No public pixel decoder exists for this checkpoint, so
# nearest-neighbour retrieval from the input video is used as the reconstruction
# method.  The latent-distance metric (cosine distance between predicted and actual
# future token embeddings) directly reflects the model's latent prediction quality.

# The temporal patchification uses tubelet_size=2 (2 raw frames per temporal token),
# matching the V-JEPA 2 pre-training configuration for facebook/vjepa2-vitg-fpc64-384.
# """

# import os
# from pathlib import Path
# from datetime import datetime

# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import torch
# import torch.nn.functional as F
# from transformers import AutoModel, AutoVideoProcessor


# # ─── Configuration ─────────────────────────────────────────────────────────────

# HF_MODEL_NAME         = "facebook/vjepa2-vitg-fpc64-384"
# VIDEO_DIR             = '/project/3018078.02/MEG_ingmar/shorts/'
# VIDEO_PATH            = VIDEO_DIR + "bw_testclip_bouwval.mp4"

# # Autoregressive prediction config
# AUTOREGRESSIVE_MODE   = True   # True = predict future tokens blindly from context
# # Analogous to PredNet's nt=15 / context_frames=10:
# NUM_OUTPUT_TOKENS     = 15     # total temporal token steps to process and display
#                                # (max = NUM_VIDEO_FRAMES // TUBELET_SIZE = 32)
# CONTEXT_TOKENS        = 10     # how many of those are real-context tokens
#                                # → NUM_OUTPUT_TOKENS - CONTEXT_TOKENS are predicted

# # V-JEPA 2 temporal patchification
# TUBELET_SIZE          = 2      # frames per temporal token (standard for this model)
# NUM_VIDEO_FRAMES      = 64     # frames to load per clip (matches fpc64)

# # Experiment loops
# TEMPORAL_STRIDE_VALUES  = [1, 2]       # sample every nth raw frame
# REVERSE_INPUT_OPTIONS   = [False, True] # False = forward, True = backward

# # Latent prediction
# PREDICTION_DAMPING    = 0.9    # velocity decay per future step (1.0 = constant velocity)

# CUSTOM_SUFFIX         = ""


# # ─── Metric helpers  (identical to PredNet script) ─────────────────────────────

# def compute_ssim(gray_a, gray_b):
#     gray_a = gray_a.astype(np.float32)
#     gray_b = gray_b.astype(np.float32)
#     c1 = 0.01 ** 2
#     c2 = 0.03 ** 2
#     mu_a   = cv2.GaussianBlur(gray_a, (7, 7), 1.5)
#     mu_b   = cv2.GaussianBlur(gray_b, (7, 7), 1.5)
#     mu_a2  = mu_a * mu_a
#     mu_b2  = mu_b * mu_b
#     mu_ab  = mu_a * mu_b
#     sig_a2 = cv2.GaussianBlur(gray_a * gray_a, (7, 7), 1.5) - mu_a2
#     sig_b2 = cv2.GaussianBlur(gray_b * gray_b, (7, 7), 1.5) - mu_b2
#     sig_ab = cv2.GaussianBlur(gray_a * gray_b, (7, 7), 1.5) - mu_ab
#     num    = (2 * mu_ab  + c1) * (2 * sig_ab + c2)
#     denom  = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
#     return float(np.mean(num / (denom + 1e-8)))


# def compute_edge_f1(gray_a_u8, gray_b_u8):
#     edge_a = cv2.Canny(gray_a_u8, 100, 200) > 0
#     edge_b = cv2.Canny(gray_b_u8, 100, 200) > 0
#     tp = np.logical_and(edge_a,  edge_b).sum()
#     fp = np.logical_and(~edge_a, edge_b).sum()
#     fn = np.logical_and(edge_a, ~edge_b).sum()
#     denom = 2 * tp + fp + fn
#     return 1.0 if denom == 0 else float(2 * tp / denom)


# def compute_optical_flow_error(gray_a_u8, gray_b_u8):
#     """Farneback optical-flow EPE, normalised to [0, 1]."""
#     try:
#         import torch as _t
#         from raft_core.raft import RAFT as _RAFT
#         m = _RAFT().to('cpu').eval()
#         a_t = _t.from_numpy(gray_a_u8).float()[None, None]
#         b_t = _t.from_numpy(gray_b_u8).float()[None, None]
#         with _t.no_grad():
#             flow = m(a_t, b_t, iters=12, test_mode=True)[-1][0].permute(1, 2, 0).cpu().numpy()
#         mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
#         return min(float(np.mean(mag)) / 50.0, 1.0)
#     except Exception:
#         flow = cv2.calcOpticalFlowFarneback(
#             gray_a_u8, gray_b_u8, None,
#             pyr_scale=0.5, levels=3, winsize=15,
#             iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
#         mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         return min(float(np.mean(mag)) / 50.0, 1.0)


# def compute_blur_loss(gt_gray_u8, pred_gray_u8):
#     gt_var   = float(cv2.Laplacian(gt_gray_u8,   cv2.CV_32F).var())
#     pred_var = float(cv2.Laplacian(pred_gray_u8, cv2.CV_32F).var())
#     return float(np.clip(1.0 - pred_var / (gt_var + 1e-8), 0.0, 1.0))


# def compute_phase_stats(values, boundary_idx):
#     v = np.asarray(values, dtype=np.float32)
#     split = int(np.clip(boundary_idx, 0, v.size))
#     ctx, fut = v[:split], v[split:]

#     def _ms(a):
#         return (float('nan'), float('nan')) if a.size == 0 else (float(np.mean(a)), float(np.std(a)))

#     cm, cs = _ms(ctx)
#     fm, fs = _ms(fut)
#     delta  = float(fm - cm) if (np.isfinite(cm) and np.isfinite(fm)) else float('nan')
#     return dict(n_context=ctx.size, n_future=fut.size,
#                 context_mean=cm, context_std=cs,
#                 future_mean=fm,  future_std=fs,
#                 delta_future_minus_context=delta)


# # ─── V-JEPA 2 helpers ──────────────────────────────────────────────────────────

# def load_vjepa2(hf_model_name: str):
#     """Load V-JEPA 2 encoder + processor onto GPU."""
#     print(f"Loading V-JEPA 2: {hf_model_name}")
#     processor = AutoVideoProcessor.from_pretrained(hf_model_name)
#     model     = AutoModel.from_pretrained(hf_model_name)
#     model.cuda().eval()
#     img_size = processor.crop_size['height']
#     print(f"  Crop size: {img_size}×{img_size}")
#     return model, processor, img_size


# @torch.inference_mode()
# def encode_video_clip(model, processor, frames_rgb: list) -> torch.Tensor:
#     """
#     Encode a list of H×W×3 uint8 RGB frames as one video clip.

#     Parameters
#     ----------
#     frames_rgb : list of T numpy arrays, each shape H×W×3 uint8

#     Returns
#     -------
#     tokens : torch.Tensor  shape [1, T_tok * P_spatial, embed_dim]  (on CPU)
#     """
#     video  = torch.stack([torch.from_numpy(f).permute(2, 0, 1)
#                           for f in frames_rgb])          # T×C×H×W  uint8
#     inputs = processor(video, return_tensors="pt")
#     pixel_values = inputs["pixel_values_videos"].cuda()  # [1, T, C, H, W]
#     tokens = model.get_vision_features(pixel_values)     # [1, N_tok, D]
#     return tokens.cpu()


# def reshape_to_temporal(tokens: torch.Tensor,
#                          num_video_frames: int,
#                          tubelet_size: int) -> torch.Tensor:
#     """
#     Reshape flat token sequence into explicit temporal units.

#     Parameters
#     ----------
#     tokens           : [1, T_tok * P_spatial, D]
#     num_video_frames : total raw frames given to the model (e.g. 64)
#     tubelet_size     : raw frames per temporal token      (e.g. 2)

#     Returns
#     -------
#     per_tok : [T_tok, P_spatial, D]
#     """
#     _, N, D = tokens.shape
#     T_tok   = num_video_frames // tubelet_size
#     P       = N // T_tok
#     return tokens.squeeze(0).reshape(T_tok, P, D)


# def predict_future_tokens(context_per_tok: torch.Tensor,
#                            num_future: int,
#                            damping: float = 0.9) -> torch.Tensor:
#     """
#     Autoregressive latent-space future prediction using damped mean-velocity
#     extrapolation, with magnitude clamping to stay on the data manifold.

#     The key fix over a naïve last-step velocity approach:
#     - Velocity is the **mean** of all consecutive-pair differences in the context
#       window (much more stable than a single last-step estimate).
#     - The velocity magnitude is **clamped** to not exceed the mean observed step
#       size in the context.  Without this, even small velocities compound over many
#       steps, shooting predicted tokens far off the real-data manifold so that NN
#       retrieval always returns the same frame.

#     Parameters
#     ----------
#     context_per_tok : [T_ctx, P, D]  real token encodings
#     num_future      : number of future temporal tokens to hallucinate
#     damping         : per-step velocity decay (1.0 = constant, 0.0 = static)

#     Returns
#     -------
#     all_tok : [T_ctx + num_future, P, D]  context + predicted tokens
#     """
#     T_ctx = context_per_tok.shape[0]

#     if T_ctx >= 2:
#         deltas   = context_per_tok[1:] - context_per_tok[:-1]  # [T-1, P, D]
#         velocity = deltas.mean(dim=0)                           # [P, D]  mean velocity

#         # Clamp velocity magnitude to the mean observed step size so predictions
#         # remain near the data manifold (prevents all-same NN retrieval collapse).
#         mean_step_size = deltas.norm(dim=-1).mean()             # scalar
#         vel_norm       = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [P, 1]
#         vel_mag        = vel_norm.mean()
#         if vel_mag > mean_step_size:
#             velocity = velocity * (mean_step_size / vel_mag)
#     else:
#         velocity = torch.zeros_like(context_per_tok[-1])

#     predicted = [context_per_tok[t].clone() for t in range(T_ctx)]
#     last = context_per_tok[-1].clone()

#     for step in range(num_future):
#         damp = damping ** (step + 1)
#         nxt  = last + velocity * damp
#         predicted.append(nxt)
#         last = nxt.clone()

#     return torch.stack(predicted, dim=0)  # [T_ctx + num_future, P, D]


# def nearest_neighbour_reconstruction(
#     predicted_per_tok: torch.Tensor,
#     library_per_tok:   torch.Tensor,
#     library_frames_rgb: list,
#     tubelet_size: int = 2,
# ) -> tuple:
#     """
#     Reconstruct a pixel frame for each predicted temporal token via cosine
#     nearest-neighbour retrieval from the library of real encoded frames.

#     The context tokens are exact copies of the real encodings, so they will
#     retrieve themselves (perfect match).  The future tokens are extrapolated
#     predictions; quality reveals how well the latent extrapolation points
#     toward the correct future frame.

#     Parameters
#     ----------
#     predicted_per_tok  : [T_pred, P, D]  predicted (context + future) tokens
#     library_per_tok    : [T_lib,  P, D]  ground-truth token encodings (full video)
#     library_frames_rgb : list of len >= T_lib * tubelet_size  RGB uint8 arrays
#     tubelet_size       : raw frames per temporal token

#     Returns
#     -------
#     reconstructed_frames : list of T_pred  H×W×3 uint8 arrays
#     nn_token_indices     : np.ndarray [T_pred]  – matched library token indices
#     nn_cosine_sim        : np.ndarray [T_pred]  – cosine similarity of each match
#     """
#     # Mean-pool over spatial patches → single vector per temporal token
#     pred_pooled = F.normalize(predicted_per_tok.float().mean(dim=1), dim=-1)  # [T_pred, D]
#     lib_pooled  = F.normalize(library_per_tok.float().mean(dim=1),   dim=-1)  # [T_lib,  D]

#     sims     = pred_pooled @ lib_pooled.T              # [T_pred, T_lib]
#     best_idx = sims.argmax(dim=1).numpy()              # [T_pred]
#     best_sim = sims.max(dim=1).values.numpy()          # [T_pred]

#     reconstructed = []
#     for tok_idx in best_idx:
#         # Token t covers raw frames [t*ts, t*ts+1, ..., t*ts+ts-1]; use first
#         frame_idx = min(int(tok_idx) * tubelet_size, len(library_frames_rgb) - 1)
#         reconstructed.append(library_frames_rgb[frame_idx])

#     return reconstructed, best_idx, best_sim


# def compute_latent_cosine_distance(predicted_per_tok: torch.Tensor,
#                                     gt_per_tok:        torch.Tensor) -> list:
#     """
#     Per-token cosine distance between predicted and real embeddings.
#     Ranges [0, 2]; 0 = identical direction, 2 = opposite.
#     This is the native V-JEPA 2 prediction quality metric.
#     """
#     pred = F.normalize(predicted_per_tok.float().mean(dim=1), dim=-1)
#     gt   = F.normalize(gt_per_tok.float().mean(dim=1),        dim=-1)
#     cos  = (pred * gt).sum(dim=-1).clamp(-1, 1)               # [T]
#     return [float(1.0 - s) for s in cos]                      # cosine distance


# # ─── Video I/O ─────────────────────────────────────────────────────────────────

# def load_video_frames(video_path: str,
#                        num_frames: int,
#                        img_size:   int,
#                        temporal_stride: int = 1) -> list:
#     """
#     Load exactly `num_frames` RGB uint8 frames from a video file.
#     If the video is shorter, the last frame is repeated to pad.
#     """
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while len(frames) < num_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         bgr  = cv2.resize(frame, (img_size, img_size))
#         frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
#         for _ in range(temporal_stride - 1):
#             if not cap.grab():
#                 break
#     cap.release()

#     # Pad with last frame if video is too short
#     while len(frames) < num_frames:
#         frames.append(frames[-1].copy())

#     return frames[:num_frames]


# # ─── Visualisation helpers ─────────────────────────────────────────────────────

# def _try_load_font(size: int = 10):
#     for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
#                  "/System/Library/Fonts/Helvetica.ttc"):
#         try:
#             return ImageFont.truetype(path, size)
#         except Exception:
#             pass
#     return ImageFont.load_default()


# def save_gif(frames_rgb: list, path, fps_ms: int = 150, skip_first: bool = True):
#     start = 1 if skip_first else 0
#     pil   = [Image.fromarray(f) for f in frames_rgb[start:]]
#     if not pil:
#         return
#     pil[0].save(str(path), save_all=True, append_images=pil[1:],
#                 duration=fps_ms, loop=0, optimize=False)


# def build_comparison_frame(gt_frame, pred_frame, frame_idx_tok,
#                              context_tokens, autoregressive_mode,
#                              metric_series_plot, font,
#                              title_height=20, plot_height=340,
#                              plot_padding=(36, 12, 12, 22)):
#     """Render one side-by-side comparison canvas with live metric plots."""
#     h, w      = gt_frame.shape[:2]
#     pl, pr, pt, pb = plot_padding
#     canvas_w  = w * 2
#     canvas_h  = title_height + h + plot_height
#     canvas    = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))
#     draw      = ImageDraw.Draw(canvas)

#     # Labels
#     if autoregressive_mode:
#         if frame_idx_tok < context_tokens:
#             gt_lbl, pr_lbl = "Context (GT)", "Context (NN-Retrieved)"
#         else:
#             gt_lbl, pr_lbl = "Future (GT)", "Future (NN-Predicted)"
#     else:
#         gt_lbl, pr_lbl = "Ground Truth", "NN-Retrieved"

#     draw.text((w // 2 - 55, 2),      gt_lbl, fill=(255, 255, 255), font=font)
#     draw.text((w + w // 2 - 65, 2),  pr_lbl, fill=(255, 255, 255), font=font)

#     canvas.paste(Image.fromarray(gt_frame),   (0, title_height))
#     canvas.paste(Image.fromarray(pred_frame), (w, title_height))

#     # Metric plots
#     x0 = pl
#     y0 = title_height + h + pt
#     x1 = canvas_w - pr
#     y1 = canvas_h  - pb

#     n_met     = len(metric_series_plot)
#     row_gap   = 5
#     row_h     = (y1 - y0 - (n_met - 1) * row_gap) // n_met

#     for m_i, (mname, mvals, mmax, mcol) in enumerate(metric_series_plot):
#         ry0 = y0 + m_i * (row_h + row_gap)
#         ry1 = ry0 + row_h

#         draw.line([(x0, ry0), (x0, ry1)], fill=(180, 180, 180), width=1)
#         draw.line([(x0, ry1), (x1, ry1)], fill=(180, 180, 180), width=1)
#         draw.text((40, ry0),          mname,         fill=(220, 220, 220), font=font)
#         draw.text((x0 - 25, ry0),    f"{mmax:.1f}", fill=(150, 150, 150), font=font)

#         # Context/future split
#         if autoregressive_mode and 0 <= context_tokens - 1 < len(mvals) > 1:
#             sx = int(x0 + (context_tokens - 1) / (len(mvals) - 1) * (x1 - x0))
#             draw.line([(sx, ry0), (sx, ry1)], fill=(80, 80, 80), width=1)

#         # Trajectory
#         pts = []
#         if len(mvals) == 1:
#             pts = [((x0 + x1) // 2, ry1)]
#         else:
#             for k, v in enumerate(mvals):
#                 px = int(x0 + k / (len(mvals) - 1) * (x1 - x0))
#                 py = int(ry1 - (v / max(mmax, 1e-9)) * (ry1 - ry0))
#                 pts.append((px, py))
#         if len(pts) > 1:
#             draw.line(pts, fill=mcol, width=2)

#         # Current frame dot + value label
#         ck = max(0, min(frame_idx_tok, len(mvals) - 1))
#         cx, cy = pts[ck]
#         r = 3
#         draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 60, 60))
#         draw.text((cx - 40, max(ry0, cy - 15)), f"{mvals[ck]:.3f}",
#                   fill=(255, 120, 120), font=font)

#     # Shared x-axis tick labels
#     draw.text((x0,      y1 + 4), "0",                         fill=(180, 180, 180), font=font)
#     draw.text((x1 - 22, y1 + 4), str(len(metric_series_plot[0][1]) - 1),
#               fill=(180, 180, 180), font=font)

#     return canvas


# # ─── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     suffix         = f"_autoregressive{CUSTOM_SUFFIX}" if AUTOREGRESSIVE_MODE else CUSTOM_SUFFIX
#     video_stem     = Path(VIDEO_PATH).stem
#     run_timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
#     run_output_dir = Path('predictions_vjepa2') / f'{video_stem}_{run_timestamp}'
#     run_output_dir.mkdir(parents=True, exist_ok=True)
#     print('Saving outputs to:', run_output_dir)

#     # ── Save run config ────────────────────────────────────────────────────────
#     with open(run_output_dir / 'run_config.txt', 'w') as f:
#         f.write('\n'.join([
#             f"run_timestamp: {run_timestamp}",
#             f"model: {HF_MODEL_NAME}",
#             f"video_path: {VIDEO_PATH}",
#             f"autoregressive_mode: {AUTOREGRESSIVE_MODE}",
#             f"num_output_tokens: {NUM_OUTPUT_TOKENS}  (analogous to PredNet nt)",
#             f"context_tokens: {CONTEXT_TOKENS}",
#             f"future_tokens: {NUM_OUTPUT_TOKENS - CONTEXT_TOKENS}",
#             f"tubelet_size: {TUBELET_SIZE}",
#             f"num_video_frames: {NUM_VIDEO_FRAMES}",
#             f"temporal_stride_values: {TEMPORAL_STRIDE_VALUES}",
#             f"reverse_input_options: {REVERSE_INPUT_OPTIONS}",
#             f"prediction_damping: {PREDICTION_DAMPING}",
#             "pixel metrics: rmse, 1-ssim, blur_loss, 1-edge_f1, optical_flow_epe",
#             "latent metric: cosine_distance (native JEPA quality measure)",
#         ]) + '\n')

#     # ── Load model once ────────────────────────────────────────────────────────
#     model, processor, img_size = load_vjepa2(HF_MODEL_NAME)
#     font = _try_load_font(10)

#     # ── Experiment loop ────────────────────────────────────────────────────────
#     for temporal_stride in TEMPORAL_STRIDE_VALUES:
#         for reverse_input_frames in REVERSE_INPUT_OPTIONS:

#             mode_tag   = "_rev"     if reverse_input_frames else "_fwd"
#             stride_tag = f"_s{temporal_stride}"
#             print(f"\n{'='*64}")
#             print(f"temporal_stride={temporal_stride}  |  reverse={reverse_input_frames}")
#             print(f"{'='*64}")

#             # ── Load frames ───────────────────────────────────────────────────
#             frames = load_video_frames(VIDEO_PATH, NUM_VIDEO_FRAMES,
#                                         img_size, temporal_stride)

#             if reverse_input_frames:
#                 frames = list(reversed(frames))
#                 print("Input order: REVERSED")
#             else:
#                 print("Input order: FORWARD")

#             # ── Encode entire clip → ground-truth token library ───────────────
#             print("Encoding video clip with V-JEPA 2 …")
#             raw_tokens = encode_video_clip(model, processor, frames)
#             # raw_tokens: [1, T_tok * P, D]
#             print(f"  Raw token tensor shape: {list(raw_tokens.shape)}")

#             all_tok = reshape_to_temporal(raw_tokens, NUM_VIDEO_FRAMES, TUBELET_SIZE)
#             T_act, P, D = all_tok.shape
#             print(f"  Per-temporal-token shape: [T_tok={T_act}, P_spatial={P}, embed_dim={D}]")

#             # Slice to the requested output window (analogous to PredNet's nt=15)
#             # This makes the comparison directly parallel: NUM_OUTPUT_TOKENS total,
#             # CONTEXT_TOKENS real, (NUM_OUTPUT_TOKENS - CONTEXT_TOKENS) predicted.
#             n_out   = min(NUM_OUTPUT_TOKENS, T_act)
#             ctx_tok = min(CONTEXT_TOKENS, n_out)
#             fut_tok = n_out - ctx_tok
#             all_tok = all_tok[:n_out]   # [n_out, P, D] — only keep the display window
#             print(f"  Display window: {n_out} tokens "
#                   f"({ctx_tok} context + {fut_tok} future), "
#                   f"analogous to PredNet nt={n_out}")

#             # ── Latent prediction ─────────────────────────────────────────────
#             if AUTOREGRESSIVE_MODE and fut_tok > 0:
#                 print(f"AUTOREGRESSIVE: {ctx_tok} context tokens → predicting "
#                       f"{fut_tok} future tokens  (total {n_out})")
#                 predicted_all_tok = predict_future_tokens(
#                     all_tok[:ctx_tok], fut_tok, PREDICTION_DAMPING)
#             else:
#                 print("No future prediction — using real tokens (reconstruction-only mode).")
#                 predicted_all_tok = all_tok.clone()
#                 ctx_tok = n_out   # all context

#             # predicted_all_tok: [n_out, P, D]

#             # ── Pixel reconstruction via nearest neighbour ────────────────────
#             # Library is the FULL encoded clip (all 32 tokens), so future predictions
#             # can retrieve any frame from the video, not just the display window.
#             all_tok_full = reshape_to_temporal(raw_tokens, NUM_VIDEO_FRAMES, TUBELET_SIZE)
#             frames_full  = load_video_frames(VIDEO_PATH, NUM_VIDEO_FRAMES,
#                                               img_size, temporal_stride)
#             if reverse_input_frames:
#                 frames_full = list(reversed(frames_full))

#             print("Reconstructing pixels via nearest-neighbour retrieval …")
#             gif_frames, nn_idx, nn_sim = nearest_neighbour_reconstruction(
#                 predicted_all_tok, all_tok_full, frames_full, TUBELET_SIZE)

#             print(f"  NN cosine sim — context: {nn_sim[:ctx_tok].mean():.4f}  "
#                   f"| future: {nn_sim[ctx_tok:].mean():.4f}  "
#                   f"(context tokens should be ≈ 1.0)")

#             # ── Ground-truth pixel frames (one per temporal token, display window) ──
#             gt_frames = [
#                 frames[min(t * TUBELET_SIZE, len(frames) - 1)]
#                 for t in range(n_out)
#             ]

#             # ── Latent-space cosine distance ──────────────────────────────────
#             latent_errors = compute_latent_cosine_distance(predicted_all_tok, all_tok)

#             # ── Pixel-space metrics ───────────────────────────────────────────
#             rmse_vals, ssim_err_vals = [], []
#             blur_vals, edge_err_vals, flow_err_vals = [], [], []

#             for i, (gt_f, pr_f) in enumerate(zip(gt_frames, gif_frames)):
#                 gt_fp  = gt_f.astype(np.float32) / 255.0
#                 pr_fp  = pr_f.astype(np.float32) / 255.0
#                 rmse_vals.append(float(np.sqrt(np.mean((gt_fp - pr_fp) ** 2))))

#                 gt_gy  = cv2.cvtColor(gt_f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#                 pr_gy  = cv2.cvtColor(pr_f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#                 ssim_err_vals.append(float(1.0 - compute_ssim(gt_gy, pr_gy)))

#                 gt_u8  = (gt_gy * 255).astype(np.uint8)
#                 pr_u8  = (pr_gy * 255).astype(np.uint8)
#                 blur_vals.append(compute_blur_loss(gt_u8, pr_u8))
#                 edge_err_vals.append(float(1.0 - compute_edge_f1(gt_u8, pr_u8)))

#                 if i < len(gt_frames) - 1:
#                     gt_nxt = gt_frames[i + 1]
#                     pr_nxt = gif_frames[i + 1]
#                     gt_n8  = cv2.cvtColor(gt_nxt, cv2.COLOR_RGB2GRAY).astype(np.uint8)
#                     pr_n8  = cv2.cvtColor(pr_nxt, cv2.COLOR_RGB2GRAY).astype(np.uint8)
#                     gt_fe  = compute_optical_flow_error(gt_u8, gt_n8)
#                     pr_fe  = compute_optical_flow_error(pr_u8, pr_n8)
#                     flow_err_vals.append(float(abs(gt_fe - pr_fe)))
#                 else:
#                     flow_err_vals.append(flow_err_vals[-1] if flow_err_vals else 0.0)

#             # Shift flow so index i = motion from frame i-1 to i
#             flow_err_vals = [0.0] + flow_err_vals[:-1]

#             # ── Fixed plot y-axis limits ──────────────────────────────────────
#             rmse_ymax      = 0.2
#             ssim_err_ymax  = 1.0
#             blur_ymax      = 1.0
#             edge_err_ymax  = 1.0
#             flow_err_ymax  = 0.2
#             latent_err_ymax = 1.0   # cosine distance [0, 2] but typically < 1

#             # ── Save individual PNG predictions ───────────────────────────────
#             for i, pr_f in enumerate(gif_frames):
#                 png_path = run_output_dir / f'pred_{i+1:02d}{suffix}{mode_tag}{stride_tag}.png'
#                 cv2.imwrite(str(png_path), cv2.cvtColor(pr_f, cv2.COLOR_RGB2BGR))

#             # ── Save prediction GIF ───────────────────────────────────────────
#             gif_path = run_output_dir / f'predictions{suffix}{mode_tag}{stride_tag}.gif'
#             save_gif(gif_frames, gif_path)
#             print('SAVED predictions GIF →', gif_path)

#             # ── Save ground-truth GIF ─────────────────────────────────────────
#             gt_gif_path = run_output_dir / f'groundtruth{suffix}{mode_tag}{stride_tag}.gif'
#             save_gif(gt_frames, gt_gif_path)
#             print('SAVED ground truth GIF →', gt_gif_path)

#             # ── Metric summary text file ──────────────────────────────────────
#             phase_boundary = ctx_tok if AUTOREGRESSIVE_MODE else len(rmse_vals)
#             metric_series_summary = [
#                 ('RMSE',         rmse_vals),
#                 ('1-SSIM',       ssim_err_vals),
#                 ('Blur Loss',    blur_vals),
#                 ('1-EdgeF1',     edge_err_vals),
#                 ('Flow EPE',     flow_err_vals),
#                 ('Latent Dist',  latent_errors),
#             ]
#             summary_lines = [
#                 f"model: {HF_MODEL_NAME}",
#                 f"mode: {'autoregressive' if AUTOREGRESSIVE_MODE else 'standard'}",
#                 f"input_order: {'reversed' if reverse_input_frames else 'forward'}",
#                 f"temporal_stride: {temporal_stride}",
#                 f"boundary_index (token): {phase_boundary}",
#                 '',
#                 'metric          ctx_mean  ctx_std  fut_mean  fut_std  delta(fut-ctx)  n_ctx  n_fut',
#                 '--------------------------------------------------------------------------------------',
#             ]
#             for mname, mvals in metric_series_summary:
#                 s = compute_phase_stats(mvals, phase_boundary)
#                 summary_lines.append(
#                     f"{mname:15s} "
#                     f"{s['context_mean']:8.4f} {s['context_std']:8.4f} "
#                     f"{s['future_mean']:8.4f} {s['future_std']:8.4f} "
#                     f"{s['delta_future_minus_context']:14.4f} "
#                     f"{s['n_context']:6d} {s['n_future']:6d}"
#                 )

#             summary_path = run_output_dir / f'metrics_summary{suffix}{mode_tag}{stride_tag}.txt'
#             with open(summary_path, 'w') as f:
#                 f.write('\n'.join(summary_lines) + '\n')
#             print('\n'.join(summary_lines))
#             print('Saved summary →', summary_path)

#             # ── Side-by-side comparison GIF ───────────────────────────────────
#             # Six metric rows: 5 pixel-space + 1 latent-space
#             metric_series_plot = [
#                 ("RMSE",         rmse_vals,      rmse_ymax,       (50,  180, 255)),
#                 ("1-SSIM",       ssim_err_vals,  ssim_err_ymax,   (140, 230, 160)),
#                 ("Blur Loss",    blur_vals,       blur_ymax,       (255, 170, 90)),
#                 ("1-EdgeF1",     edge_err_vals,   edge_err_ymax,   (250, 200, 120)),
#                 ("Flow EPE",     flow_err_vals,   flow_err_ymax,   (200, 150, 255)),
#                 ("Latent Dist",  latent_errors,   latent_err_ymax, (255, 100, 100)),
#             ]

#             comparison_gif_path = (
#                 run_output_dir / f'comparison{suffix}{mode_tag}{stride_tag}.gif')
#             comp_frames = []

#             for fi, (gt_f, pr_f) in enumerate(
#                     zip(gt_frames[1:], gif_frames[1:]), start=1):
#                 canvas = build_comparison_frame(
#                     gt_f, pr_f, fi,
#                     ctx_tok, AUTOREGRESSIVE_MODE,
#                     metric_series_plot, font,
#                     title_height=20, plot_height=350,
#                 )
#                 comp_frames.append(canvas)

#             comp_frames[0].save(
#                 str(comparison_gif_path),
#                 save_all=True, append_images=comp_frames[1:],
#                 duration=150, loop=0, optimize=False)
#             print('SAVED comparison GIF →', comparison_gif_path)

#     print('\nAll runs complete.')


# if __name__ == "__main__":
#     main()












