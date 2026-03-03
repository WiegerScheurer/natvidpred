#!/usr/bin/env python3
"""
vidpred_vjepa2_vitg.py
======================
Video next-frame prediction using V-JEPA 2 (ViT-G encoder) + the jepa-wms
pixel decoder, mirroring vidpred_prednet.py in structure and outputs.

─── Setup you need before running ───────────────────────────────────────────

1.  Clone jepa-wms and point JEPA_WMS_REPO_DIR below at it:
      git clone https://github.com/facebookresearch/jepa-wms
      # The script adds  jepa-wms/  to sys.path automatically.

2.  Download the decoder weights (one file, ~1.1 GB):
      curl -O https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_vj2vitgnorm_vitldec_dup_256_INet.pth.tar
      # Point DECODER_CKPT_PATH below at the downloaded file.

3.  Make sure facebook/vjepa2-vitg-fpc64-256 is accessible via HuggingFace
    (it will be downloaded to HF_CACHE on first run).

─── Why ViT-G (not ViT-L)? ──────────────────────────────────────────────────
    The decoder weights were trained against V-JEPA2 ViT-G encoder output
    (embed_dim = 1408).  ViT-L produces 1024-dim tokens — dimension mismatch.
    Using ViT-G avoids any projection layer and gives the decoder exactly
    what it was trained on.

─── Why fpc64-256 (not fpc64-384)? ─────────────────────────────────────────
    The decoder was trained at 256 px output, which corresponds to a
    (256 / 16)^2 = 256 spatial-patch layout.  The 384-px model would produce
    576 spatial patches — wrong shape for the decoder.  Use 256-px here.

─── Why the old script froze and went dark ──────────────────────────────────
    The old script mean-pooled ALL patch tokens into one global vector per
    frame.  After 1-2 autoregressive steps the extrapolated global vector
    drifts off the manifold; cosine similarities to all context vectors
    become nearly equal; softmax gives near-uniform blend weights;
    output = average of all context frames = blurry, and progressively
    darker as more frames accumulate.

    Fix: keep FULL spatial patch tokens  (N_patches, D) = (256, 1408) per
    frame.  Extrapolate per-patch, decode spatially with the ViT decoder.

─── Prediction pipeline ─────────────────────────────────────────────────────
    1.  Encode every context frame → (N_patches, D) spatial token matrix.
    2.  Per-patch linear velocity extrapolation:
            delta[p] = tokens[-1][p] - tokens[-2][p]
            pred[p]  = tokens[-1][p] + steps * delta[p]
    3.  Decode predicted token matrix → (H, W, 3) pixels  via jepa-wms decoder.

─── Fallback (no decoder / load fails) ─────────────────────────────────────
    Per-patch nearest-neighbour blending:
    For each spatial position p, compute cosine sim of pred[p] against every
    context token at that position, then blend the context pixels weighted by
    softmax(sim / tau).  This is spatially local and avoids the global-collapse
    problem that caused the darkening artefact.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

# ViT-G model — matches decoder's expected token dimensionality (1408)
# and spatial layout (256 patches @ 256 px with patch_size=16)
HF_REPO         = "facebook/vjepa2-vitg-fpc64-256"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# jepa-wms repo root (after: git clone https://github.com/facebookresearch/jepa-wms)
JEPA_WMS_REPO_DIR  = "/project/3018078.02/natvidpred_workspace/jepa-wms"

# Decoder checkpoint downloaded from Meta's CDN
DECODER_CKPT_PATH  = "/project/3018078.02/natvidpred_workspace/vm2m_lpips_vj2vitgnorm_vitldec_dup_256_INet.pth.tar"

VIDEO_DIR       = "/project/3018078.02/MEG_ingmar/shorts/"
VIDEO_PATH      = VIDEO_DIR + "bw_testclip_bouwval.mp4"

CONTEXT_FRAMES  = 10        # frames used as real context
TOTAL_FRAMES    = 15        # context + frames to predict
TEMPORAL_STRIDE = 1

# Spatial resolution the decoder was trained on — keep at 256
IMG_SIZE        = 256
PATCH_SIZE      = 16                    # ViT-G uses 16-px patches
N_PATCHES       = (IMG_SIZE // PATCH_SIZE) ** 2    # 256

# fpc64 model: 64-frame clips; tubelets are 2 frames deep
N_CLIP_FRAMES   = 64
TUBELET_DEPTH   = 2                     # frames per tubelet

# Fallback blending temperature (used when decoder is unavailable)
BLEND_TEMPERATURE = 0.07

AUTOREGRESSIVE_MODE = True

# Fixed y-axis maxima for metric plots
RMSE_PLOT_MAX      = 0.20
SSIM_ERR_PLOT_MAX  = 1.00
BLUR_LOSS_PLOT_MAX = 1.00
EDGE_ERR_PLOT_MAX  = 1.00
FLOW_ERR_PLOT_MAX  = 0.20

# ─────────────────────────────────────────────────────────────────────────────
#  METRIC HELPERS  (unchanged from vidpred_prednet.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(gray_a, gray_b):
    a, b = gray_a.astype(np.float32), gray_b.astype(np.float32)
    c1, c2 = 0.01**2, 0.03**2
    ma, mb = cv2.GaussianBlur(a,(7,7),1.5), cv2.GaussianBlur(b,(7,7),1.5)
    ma2,mb2,mab = ma*ma, mb*mb, ma*mb
    sa2 = cv2.GaussianBlur(a*a,(7,7),1.5)-ma2
    sb2 = cv2.GaussianBlur(b*b,(7,7),1.5)-mb2
    sab = cv2.GaussianBlur(a*b,(7,7),1.5)-mab
    return float(np.mean(((2*mab+c1)*(2*sab+c2))/((ma2+mb2+c1)*(sa2+sb2+c2)+1e-8)))

def compute_edge_f1(a_u8, b_u8):
    ea = cv2.Canny(a_u8,100,200)>0;  eb = cv2.Canny(b_u8,100,200)>0
    tp=np.logical_and(ea,eb).sum(); fp=np.logical_and(~ea,eb).sum(); fn=np.logical_and(ea,~eb).sum()
    d=2*tp+fp+fn;  return 1.0 if d==0 else float(2*tp/d)

def compute_optical_flow_error(a_u8, b_u8):
    flow=cv2.calcOpticalFlowFarneback(a_u8,b_u8,None,0.5,3,15,3,5,1.2,0)
    mag,_=cv2.cartToPolar(flow[...,0],flow[...,1])
    return float(min(np.mean(mag)/50.0,1.0))

def compute_blur_loss(gt_u8, pred_u8):
    gv=float(cv2.Laplacian(gt_u8,  cv2.CV_32F).var())
    pv=float(cv2.Laplacian(pred_u8,cv2.CV_32F).var())
    return float(np.clip(1.0-pv/(gv+1e-8),0.0,1.0))

def compute_phase_stats(values, boundary_idx):
    v=np.asarray(values,dtype=np.float32); b=int(np.clip(boundary_idx,0,v.size))
    ctx,fut=v[:b],v[b:]
    def ms(a): return (float('nan'),float('nan')) if a.size==0 else (float(np.mean(a)),float(np.std(a)))
    cm,cs=ms(ctx); fm,fs=ms(fut)
    delta=float(fm-cm) if np.isfinite(cm) and np.isfinite(fm) else float('nan')
    return dict(n_context=int(ctx.size),n_future=int(fut.size),
                context_mean=cm,context_std=cs,future_mean=fm,future_std=fs,
                delta_future_minus_context=delta)


# ─────────────────────────────────────────────────────────────────────────────
#  DECODER LOADING
# ─────────────────────────────────────────────────────────────────────────────

def try_load_decoder(ckpt_path: str, repo_dir: str, device: str):
    """
    Load the jepa-wms ViT-L decoder from a checkpoint.

    The checkpoint is a .pth.tar file.  We try several common key layouts:
      - ckpt['decoder']  (most likely from jepa-wms training code)
      - ckpt['model']
      - ckpt itself if it's a state_dict

    The decoder class is imported from jepa-wms/src/models/ — we try a few
    plausible class names.  If nothing works we return None and the script
    falls back to per-patch cosine blending.

    Returns (decoder_module, input_proj) or (None, None).
      input_proj: optional nn.Linear(1408, decoder_input_dim) if dims differ.
    """
    import torch.nn as nn

    if not Path(ckpt_path).exists():
        print(f"[decoder] Checkpoint not found: {ckpt_path}  → using fallback blending")
        return None, None

    if not Path(repo_dir).exists():
        print(f"[decoder] jepa-wms repo not found: {repo_dir}  → using fallback blending")
        return None, None

    # Add jepa-wms source to path
    for sub in ["", "src", os.path.join("src","models")]:
        p = str(Path(repo_dir) / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

    print(f"[decoder] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ── Extract state dict ──────────────────────────────────────────────
    decoder_state = None
    for key in ("decoder", "model", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            decoder_state = ckpt[key]
            print(f"[decoder] Found state_dict under key '{key}'")
            break
    if decoder_state is None and isinstance(ckpt, dict):
        decoder_state = ckpt   # assume the ckpt IS the state dict
        print("[decoder] Using top-level checkpoint as state dict")

    # ── Try to import decoder class ─────────────────────────────────────
    decoder_cls = None
    import_attempts = [
        ("models.decoder",   ["VideoDecoder","ViTDecoder","Decoder","VisionTransformerDecoder"]),
        ("decoder",          ["VideoDecoder","ViTDecoder","Decoder","VisionTransformerDecoder"]),
        ("vit",              ["VisionTransformerDecoder"]),
        ("vision_transformer",["VisionTransformerDecoder"]),
    ]
    for module_name, class_names in import_attempts:
        try:
            mod = __import__(module_name, fromlist=class_names)
            for cls_name in class_names:
                if hasattr(mod, cls_name):
                    decoder_cls = getattr(mod, cls_name)
                    print(f"[decoder] Imported {cls_name} from {module_name}")
                    break
            if decoder_cls is not None:
                break
        except ImportError:
            continue

    if decoder_cls is None:
        print("[decoder] Could not import decoder class from jepa-wms — using fallback blending")
        return None, None

    # ── Instantiate decoder ─────────────────────────────────────────────
    # Try common constructor signatures; adjust if your build differs.
    decoder = None
    init_attempts = [
        # (kwargs_dict, description)
        (dict(embed_dim=1024, depth=8, num_heads=16,
              mlp_ratio=4.0, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
              in_chans=3, num_patches=N_PATCHES),
         "ViT-L decoder standard config"),
        (dict(embed_dim=1024, depth=8, num_heads=16, patch_size=PATCH_SIZE,
              img_size=IMG_SIZE, num_patches=N_PATCHES),
         "ViT-L decoder reduced config"),
        (dict(),
         "no-arg constructor"),
    ]
    for kwargs, desc in init_attempts:
        try:
            decoder = decoder_cls(**kwargs)
            print(f"[decoder] Instantiated with: {desc}")
            break
        except Exception as e:
            print(f"[decoder] Constructor {desc} failed: {e}")

    if decoder is None:
        print("[decoder] All constructor attempts failed — using fallback blending")
        return None, None

    # ── Load weights ────────────────────────────────────────────────────
    try:
        missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
        print(f"[decoder] Weights loaded  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
    except Exception as e:
        print(f"[decoder] load_state_dict failed: {e}  → using fallback blending")
        return None, None

    decoder.eval().to(device)

    # ── Projection layer (ViT-G 1408 → decoder input dim) ───────────────
    # Inspect the decoder's first linear layer to find its expected input dim.
    decoder_input_dim = None
    for name, param in decoder.named_parameters():
        if "weight" in name and param.ndim == 2:
            decoder_input_dim = param.shape[1]
            print(f"[decoder] First weight param '{name}': shape={tuple(param.shape)}  "
                  f"→ decoder expects input_dim={decoder_input_dim}")
            break

    VITG_DIM = 1408
    if decoder_input_dim is not None and decoder_input_dim != VITG_DIM:
        print(f"[decoder] Adding projection: {VITG_DIM} → {decoder_input_dim}")
        import torch.nn as nn
        proj = nn.Linear(VITG_DIM, decoder_input_dim, bias=False).to(device)
        # Projection weights are randomly initialised; for now we use them as-is.
        # To get better results, fine-tune proj on a small video dataset.
        proj.eval()
    else:
        proj = None

    print("[decoder] Decoder ready.")
    return decoder, proj


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_backbone(model):
    for attr in ("vjepa2","model","encoder","backbone","vision_model"):
        if hasattr(model, attr):
            sub = getattr(model, attr)
            if hasattr(sub, "parameters"):
                return sub
    named = [(n, type(m).__name__) for n,m in model.named_children()]
    raise AttributeError(f"Cannot find backbone. Children: {named}")


def frame_to_clip_tensor(frame_bgr, processor, device):
    """
    Tile a single BGR frame into a (1, T=64, C, H, W) clip tensor.
    Processor expects (T, H, W, C) uint8 numpy, returns 'pixel_values_videos'.
    """
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    clip   = np.stack([rgb] * N_CLIP_FRAMES, axis=0)   # (64, H, W, 3)
    inputs = processor(clip, return_tensors="pt")
    pv     = inputs["pixel_values_videos"]              # (1, 64, C, H, W)
    while pv.ndim < 5:
        pv = pv.unsqueeze(0)
    return pv.to(device)


@torch.no_grad()
def encode_frame_full_tokens(frame_bgr, backbone, processor, device):
    """
    Encode one BGR frame, returning FULL spatial patch tokens — NOT mean-pooled.

    V-JEPA2 processes 64-frame clips with tubelet_depth=2, so the backbone
    produces N_temporal × N_spatial tokens:
        N_temporal = 64 / tubelet_depth = 32
        N_spatial  = (IMG_SIZE / PATCH_SIZE)^2 = 256
        Total      = 32 × 256 = 8192 tokens  (shape: [1, 8192, 1408])

    Because all 64 frames in the clip are identical (one frame tiled), the
    tokens at every temporal position are nearly identical. We extract the
    tokens corresponding to the MIDDLE temporal position so the positional
    encoding is well-centred.

    Returns: Tensor of shape (N_spatial, D) = (256, 1408) on CPU.
    """
    pv  = frame_to_clip_tensor(frame_bgr, processor, device)
    out = backbone(pixel_values_videos=pv)

    hidden = out.last_hidden_state if hasattr(out,"last_hidden_state") else out[0]
    hidden = hidden.squeeze(0)                          # (N_total_tokens, D)

    # Strip CLS token if present: detect by checking if total count is
    # N_temporal*N_spatial+1 (with CLS) or N_temporal*N_spatial (without).
    n_temporal = N_CLIP_FRAMES // TUBELET_DEPTH         # 32
    n_total_expected = n_temporal * N_PATCHES           # 8192
    if hidden.shape[0] == n_total_expected + 1:
        hidden = hidden[1:]                             # remove CLS

    if hidden.shape[0] != n_total_expected:
        print(f"[encode] Unexpected token count {hidden.shape[0]} "
              f"(expected {n_total_expected}). Using all tokens mean-pooled "
              f"per spatial position as fallback.")
        # If shape is unexpected, fall back to mean over temporal dim
        # by just taking every N_patches-th group
        n_avail = hidden.shape[0]
        if n_avail >= N_PATCHES:
            hidden = hidden[:N_PATCHES]
        else:
            # Truly unexpected — just repeat to fill
            repeats = (N_PATCHES + n_avail - 1) // n_avail
            hidden  = hidden.repeat(repeats, 1)[:N_PATCHES]
    else:
        # Take the middle temporal slice
        mid_t   = n_temporal // 2                       # 16
        start   = mid_t * N_PATCHES
        hidden  = hidden[start : start + N_PATCHES]     # (256, 1408)

    return hidden.cpu()                                 # (N_patches=256, D=1408)


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION  (per-patch linear velocity)
# ─────────────────────────────────────────────────────────────────────────────

def predict_tokens_linear(context_token_list, steps_ahead=1):
    """
    Per-patch linear velocity extrapolation.

    context_token_list: list of (N_patches, D) tensors
    Returns: (N_patches, D) predicted token matrix (NOT normalised — raw space)
    """
    if len(context_token_list) < 2:
        return context_token_list[-1].clone()
    velocity = context_token_list[-1] - context_token_list[-2]   # (N_patches, D)
    return context_token_list[-1] + steps_ahead * velocity


# ─────────────────────────────────────────────────────────────────────────────
#  PIXEL RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_tokens_with_decoder(pred_tokens, decoder, proj, device):
    """
    Decode (N_patches, D) token matrix to a (H, W, 3) uint8 RGB frame
    using the jepa-wms ViT-L decoder.

    The decoder likely expects (B, N_patches, D) and outputs one of:
      - (B, 3, H, W)  directly
      - (B, N_patches, patch_h*patch_w*3)  → needs unpatchify

    We try both and pick whichever succeeds.
    """
    tokens = pred_tokens.unsqueeze(0).to(device)        # (1, 256, 1408)

    if proj is not None:
        tokens = proj(tokens)                           # (1, 256, decoder_dim)

    out = decoder(tokens)                               # try forward

    # ── Interpret output ────────────────────────────────────────────────
    if out.ndim == 4:
        # (B, C, H, W) — already spatial
        img = out[0].permute(1,2,0).cpu().numpy()       # (H, W, C)
    elif out.ndim == 3:
        # (B, N_patches, patch_h*patch_w*3) — unpatchify
        B, N, pw3 = out.shape
        ph = pw = PATCH_SIZE
        assert pw3 == ph * pw * 3, f"Unexpected last dim: {pw3}"
        G = int(N ** 0.5)                               # grid side = 16
        img_t = out[0].reshape(G, G, ph, pw, 3)
        img_t = img_t.permute(4, 0, 2, 1, 3)           # (3, G, ph, G, pw)
        img_t = img_t.reshape(3, G*ph, G*pw)            # (3, H, W)
        img   = img_t.permute(1,2,0).cpu().numpy()      # (H, W, 3)
    else:
        raise ValueError(f"Unexpected decoder output shape: {out.shape}")

    img = np.clip(img, 0.0, 1.0) if img.max() <= 1.1 else np.clip(img/255.0, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def decode_tokens_per_patch_blend(pred_tokens, context_token_list,
                                   context_frames_rgb, temperature=0.07):
    """
    Fallback decoder: per-patch cosine-similarity weighted blend.

    For each spatial position p:
        sim[i] = cosine_similarity(pred_tokens[p], context_tokens[i][p])
        w[i]   = softmax(sim / temperature)
        output[p_region] = sum_i w[i] * context_frames_rgb[i][patch_p_region]

    This is spatially local — each patch blends independently — so there
    is NO global collapse and no darkening.  The output shows the correct
    scene content at each location weighted by how well the prediction
    matches each context frame's token at that location.

    Returns: (H, W, 3) uint8 RGB.
    """
    G          = int(N_PATCHES ** 0.5)                   # 16 (grid side)
    P          = PATCH_SIZE                               # 16 px
    H = W      = G * P                                   # 256

    pred_norm  = F.normalize(pred_tokens, dim=-1)        # (256, D)
    ctx_norm   = [F.normalize(t, dim=-1) for t in context_token_list]

    output = np.zeros((H, W, 3), dtype=np.float32)

    for p in range(N_PATCHES):
        # Row / col of this patch in the spatial grid
        row = p // G;  col = p % G
        y0, y1 = row*P, (row+1)*P
        x0, x1 = col*P, (col+1)*P

        # Per-patch similarities: (N_context,)
        sims = torch.stack([
            torch.dot(pred_norm[p], cn[p]) for cn in ctx_norm
        ])
        weights = F.softmax(sims / temperature, dim=0).numpy()   # (N_context,)

        # Weighted blend of context patches at this position
        patch_blend = np.zeros((P, P, 3), dtype=np.float32)
        for wi, fr in zip(weights, context_frames_rgb):
            patch_blend += wi * fr[y0:y1, x0:x1].astype(np.float32)

        output[y0:y1, x0:x1] = patch_blend

    return np.clip(output, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoVideoProcessor, AutoModelForVideoClassification

    video_stem     = Path(VIDEO_PATH).stem
    run_timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path("predictions") / f"{video_stem}_vjepa2g_{run_timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print("Saving outputs to:", run_output_dir)

    suffix       = "_autoregressive" if AUTOREGRESSIVE_MODE else ""
    strategy_tag = "_fixfr"
    mode_tag     = "_fwd"
    stride_tag   = f"_s{TEMPORAL_STRIDE}"

    # ── run config ─────────────────────────────────────────────────────────
    cfg_path = run_output_dir / "run_config.txt"
    with open(cfg_path, "w") as f:
        for k, v in [("run_timestamp",run_timestamp),("hf_repo",HF_REPO),
                     ("video_path",VIDEO_PATH),("device",DEVICE),
                     ("img_size",IMG_SIZE),("patch_size",PATCH_SIZE),
                     ("n_patches",N_PATCHES),("n_clip_frames",N_CLIP_FRAMES),
                     ("context_frames",CONTEXT_FRAMES),("total_frames",TOTAL_FRAMES),
                     ("temporal_stride",TEMPORAL_STRIDE),
                     ("autoregressive_mode",AUTOREGRESSIVE_MODE),
                     ("decoder_ckpt",DECODER_CKPT_PATH),
                     ("jepa_wms_repo",JEPA_WMS_REPO_DIR)]:
            f.write(f"{k}: {v}\n")
        f.write("metrics: rmse, 1-ssim, blur_loss, 1-edge_f1, optical_flow_epe\n")
    print("Saved run config to:", cfg_path)

    # ── load decoder (before model to fail fast) ───────────────────────────
    print("\n── Loading jepa-wms decoder ──────────────────────────────────────")
    decoder, proj = try_load_decoder(DECODER_CKPT_PATH, JEPA_WMS_REPO_DIR, DEVICE)
    using_decoder = decoder is not None
    print(f"Decoder mode: {'jepa-wms ViT-L' if using_decoder else 'per-patch cosine blend (fallback)'}")

    # ── load V-JEPA2 ViT-G encoder ─────────────────────────────────────────
    print(f"\n── Loading encoder: {HF_REPO} ───────────────────────────────────")
    model     = AutoModelForVideoClassification.from_pretrained(HF_REPO).to(DEVICE)
    processor = AutoVideoProcessor.from_pretrained(HF_REPO)
    model.eval()
    backbone  = get_backbone(model)
    print(f"Encoder loaded: {type(backbone).__name__}  |  device={DEVICE}")

    # Sanity-check that we are using ViT-G (embed_dim=1408)
    try:
        embed_dim = backbone.config.hidden_size
    except Exception:
        embed_dim = "unknown"
    print(f"Encoder embed_dim: {embed_dim}  (expected 1408 for ViT-G)")

    # ── load video ─────────────────────────────────────────────────────────
    print(f"\nReading video: {VIDEO_PATH}")
    cap, raw_frames = cv2.VideoCapture(VIDEO_PATH), []
    while len(raw_frames) < TOTAL_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        raw_frames.append(frame)
        for _ in range(TEMPORAL_STRIDE - 1):
            if not cap.grab(): break
    cap.release()
    if len(raw_frames) < TOTAL_FRAMES:
        raise ValueError(f"Need {TOTAL_FRAMES} frames, got {len(raw_frames)}.")
    raw_frames = raw_frames[:TOTAL_FRAMES]
    print(f"Loaded {len(raw_frames)} frames  (stride={TEMPORAL_STRIDE})")

    # Resize BGR to IMG_SIZE for pixel metrics; keep originals for encoding
    def to_rgb(bgr):
        return cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    frames_rgb = [to_rgb(f) for f in raw_frames]

    # ── encode context frames ──────────────────────────────────────────────
    print(f"\nEncoding {CONTEXT_FRAMES} context frames (full patch tokens) ...")
    context_tokens    = []    # list of (N_patches, D) tensors
    context_frames_rgb = []

    for i in range(CONTEXT_FRAMES):
        tokens = encode_frame_full_tokens(raw_frames[i], backbone, processor, DEVICE)
        context_tokens.append(tokens)
        context_frames_rgb.append(frames_rgb[i])
        print(f"  Frame {i+1}/{CONTEXT_FRAMES}  token shape={tuple(tokens.shape)}")

    # ── predict future frames ──────────────────────────────────────────────
    n_future             = TOTAL_FRAMES - CONTEXT_FRAMES
    predicted_frames_rgb = list(context_frames_rgb)
    running_tokens       = list(context_tokens)

    print(f"\nPredicting {n_future} future frames ...")
    for step in range(n_future):
        pred_tok = predict_tokens_linear(running_tokens, steps_ahead=1)

        if using_decoder:
            try:
                frame_pred = decode_tokens_with_decoder(pred_tok, decoder, proj, DEVICE)
                mode_str   = "decoder"
            except Exception as e:
                print(f"  [step {step+1}] Decoder forward failed ({e}), using blend fallback")
                frame_pred = decode_tokens_per_patch_blend(
                    pred_tok, running_tokens, context_frames_rgb, BLEND_TEMPERATURE)
                mode_str   = "blend"
        else:
            frame_pred = decode_tokens_per_patch_blend(
                pred_tok, running_tokens, context_frames_rgb, BLEND_TEMPERATURE)
            mode_str   = "blend"

        predicted_frames_rgb.append(frame_pred)
        if AUTOREGRESSIVE_MODE:
            running_tokens.append(pred_tok)

        print(f"  Future frame {step+1}/{n_future}  [{mode_str}]")

    # ── save PNGs ──────────────────────────────────────────────────────────
    for i, fr in enumerate(predicted_frames_rgb):
        cv2.imwrite(
            str(run_output_dir / f"pred_{i+1:02d}{suffix}{strategy_tag}{mode_tag}{stride_tag}.png"),
            cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        )
    print(f"Saved {TOTAL_FRAMES} PNGs")

    # ── prediction GIF ─────────────────────────────────────────────────────
    gif_path = run_output_dir / f"predictions{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
    pil_pred = [Image.fromarray(f) for f in predicted_frames_rgb[1:]]
    pil_pred[0].save(str(gif_path), save_all=True, append_images=pil_pred[1:],
                     duration=150, loop=0, optimize=False)
    print("SAVED predictions GIF  ->", gif_path)

    # ── ground-truth GIF ───────────────────────────────────────────────────
    gt_gif = run_output_dir / f"groundtruth{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
    gt_pil = [Image.fromarray(f) for f in frames_rgb[1:]]
    gt_pil[0].save(str(gt_gif), save_all=True, append_images=gt_pil[1:],
                   duration=150, loop=0, optimize=False)
    print("SAVED ground-truth GIF ->", gt_gif)

    # ── framewise metrics ──────────────────────────────────────────────────
    print("\nComputing metrics ...")
    rmse_v, ssim_v, blur_v, edge_v, flow_v = [], [], [], [], []

    for idx, (gt_rgb, pr_rgb) in enumerate(zip(frames_rgb, predicted_frames_rgb)):
        gt_f = gt_rgb.astype(np.float32)/255.0
        pr_f = pr_rgb.astype(np.float32)/255.0
        rmse_v.append(float(np.sqrt(np.mean((gt_f-pr_f)**2))))

        gt_g = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        pr_g = cv2.cvtColor(pr_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        ssim_v.append(float(1.0-compute_ssim(gt_g, pr_g)))

        gt_u8=(gt_g*255).astype(np.uint8); pr_u8=(pr_g*255).astype(np.uint8)
        blur_v.append(compute_blur_loss(gt_u8, pr_u8))
        edge_v.append(float(1.0-compute_edge_f1(gt_u8, pr_u8)))

        if idx < len(frames_rgb)-1:
            gn=cv2.cvtColor(frames_rgb[idx+1],           cv2.COLOR_RGB2GRAY).astype(np.uint8)
            pn=cv2.cvtColor(predicted_frames_rgb[idx+1], cv2.COLOR_RGB2GRAY).astype(np.uint8)
            flow_v.append(float(abs(compute_optical_flow_error(gt_u8,gn)-
                                    compute_optical_flow_error(pr_u8,pn))))
        else:
            flow_v.append(flow_v[-1] if flow_v else 0.0)

    flow_v = [0.0] + flow_v[:-1]

    # ── metrics summary ────────────────────────────────────────────────────
    phase_b = CONTEXT_FRAMES if AUTOREGRESSIVE_MODE else len(rmse_v)
    series  = [("RMSE",rmse_v),("1-SSIM",ssim_v),("Blur Loss",blur_v),
               ("1-EdgeF1",edge_v),("Flow EPE",flow_v)]

    summary_path = run_output_dir / f"metrics_summary{suffix}{strategy_tag}{mode_tag}{stride_tag}.txt"
    lines = [
        f"model: {HF_REPO}",
        f"decoder: {'jepa-wms ViT-L' if using_decoder else 'per-patch cosine blend'}",
        f"prediction: per-patch linear velocity extrapolation",
        f"boundary_index: {phase_b}",
        "",
        "metric       ctx_mean  ctx_std  fut_mean  fut_std  delta(fut-ctx)  n_ctx  n_fut",
        "--------------------------------------------------------------------------------",
    ]
    for name, vals in series:
        s = compute_phase_stats(vals, phase_b)
        lines.append(f"{name:11s} "
                     f"{s['context_mean']:8.4f} {s['context_std']:8.4f} "
                     f"{s['future_mean']:8.4f} {s['future_std']:8.4f} "
                     f"{s['delta_future_minus_context']:14.4f} "
                     f"{s['n_context']:6d} {s['n_future']:6d}")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines)+"\n")
    print("Saved metrics summary to:", summary_path)
    print("\n".join(lines))

    # ── comparison GIF ─────────────────────────────────────────────────────
    print("\nBuilding comparison GIF ...")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except Exception:
            font = ImageFont.load_default()

    H_f, W_f    = frames_rgb[0].shape[:2]
    title_h     = 20; plot_h = 285
    pad_l=36; pad_r=12; pad_t=12; pad_b=22

    plot_metrics = [
        ("RMSE",      rmse_v, RMSE_PLOT_MAX,      ( 50,180,255)),
        ("1-SSIM",    ssim_v, SSIM_ERR_PLOT_MAX,  (140,230,160)),
        ("Blur Loss", blur_v, BLUR_LOSS_PLOT_MAX, (255,170, 90)),
        ("1-EdgeF1",  edge_v, EDGE_ERR_PLOT_MAX,  (250,200,120)),
        ("Flow EPE",  flow_v, FLOW_ERR_PLOT_MAX,  (200,150,255)),
    ]

    cmp_frames = []
    for fi, (gt_f, pr_f) in enumerate(
            zip(frames_rgb[1:], predicted_frames_rgb[1:]), start=1):

        cw=W_f*2; ch=title_h+H_f+plot_h
        cv_img=Image.new("RGB",(cw,ch),(0,0,0))
        draw=ImageDraw.Draw(cv_img)

        if AUTOREGRESSIVE_MODE and fi < CONTEXT_FRAMES:
            g_lbl, p_lbl = "Context (GT)", "Context (Pred)"
        elif AUTOREGRESSIVE_MODE:
            g_lbl, p_lbl = "Future (GT)", "Future (Pred)"
        else:
            g_lbl, p_lbl = "Ground Truth", "Prediction"

        draw.text((W_f//2-40, 2),       g_lbl, fill=(255,255,255), font=font)
        draw.text((W_f+W_f//2-30, 2),   p_lbl, fill=(255,255,255), font=font)
        cv_img.paste(Image.fromarray(gt_f), (0,   title_h))
        cv_img.paste(Image.fromarray(pr_f), (W_f, title_h))

        px0=pad_l; py0=title_h+H_f+pad_t; px1=cw-pad_r; py1=ch-pad_b
        n_m=len(plot_metrics); row_gap=5
        row_h=(py1-py0-(n_m-1)*row_gap)//n_m

        for mi,(mn,mv,mm,lc) in enumerate(plot_metrics):
            ry0=py0+mi*(row_h+row_gap); ry1=ry0+row_h
            draw.line([(px0,ry0),(px0,ry1)], fill=(180,180,180), width=1)
            draw.line([(px0,ry1),(px1,ry1)], fill=(180,180,180), width=1)
            draw.text((40,ry0), mn, fill=(220,220,220), font=font)
            draw.text((px0-25,ry0), f"{mm:.1f}", fill=(150,150,150), font=font)

            if AUTOREGRESSIVE_MODE and 0<=CONTEXT_FRAMES-1<len(mv) and len(mv)>1:
                sx=int(px0+(CONTEXT_FRAMES-1)/(len(mv)-1)*(px1-px0))
                draw.line([(sx,ry0),(sx,ry1)], fill=(80,80,80), width=1)

            pts=[(int(px0+k/(len(mv)-1)*(px1-px0)), int(ry1-(v/mm)*(ry1-ry0)))
                 for k,v in enumerate(mv)] if len(mv)>1 else [((px0+px1)//2,ry1)]

            if len(pts)>1: draw.line(pts, fill=lc, width=2)
            ck=max(0,min(fi,len(mv)-1)); cx,cy=pts[ck]
            draw.ellipse((cx-3,cy-3,cx+3,cy+3), fill=(255,60,60))
            draw.text((cx-40,max(ry0,cy-15)), f"{mv[ck]:.3f}", fill=(255,120,120), font=font)

        draw.text((px0,   py1+4), "0",                fill=(180,180,180), font=font)
        draw.text((px1-22,py1+4), str(len(rmse_v)-1), fill=(180,180,180), font=font)
        cmp_frames.append(cv_img)

    cmp_gif = run_output_dir / f"comparison{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
    cmp_frames[0].save(str(cmp_gif), save_all=True, append_images=cmp_frames[1:],
                       duration=150, loop=0, optimize=False)
    print("SAVED comparison GIF   ->", cmp_gif)
    print("\nAll done. Output directory:", run_output_dir)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# vidpred_vjepa2.py
# =================
# Video next-frame prediction using V-JEPA 2, mirroring vidpred_prednet.py.
# Same metrics (RMSE, 1-SSIM, Blur Loss, 1-EdgeF1, Optical Flow EPE),
# same GIF outputs (predictions, ground truth, side-by-side comparison),
# same run_config.txt and metrics_summary.txt.

# Bug-fixes vs. original version
# ───────────────────────────────
#   1. Processor output key is "pixel_values_videos", NOT "pixel_values".
#   2. V-JEPA2VideoProcessor expects a (T, H, W, C) uint8 numpy array,
#      NOT a list of PIL images.
#   3. VJEPA2Model forward kwarg is pixel_values_videos=, NOT pixel_values=.

# Prediction strategy  (zero-shot — no decoder training required)
# ──────────────────────────────────────────────────────────────
#   1. Encode each context frame individually via V-JEPA2's ViT backbone
#      → latent vector z_i  (mean-pooled over spatial patch tokens, L2-normed)
#   2. Compute latent velocity:  Delta = z_{n-1} - z_{n-2}
#   3. Linear extrapolation:  z_hat_{n+k} = z_{n-1} + k * Delta
#   4. Reconstruct pixels: softmax(cosine_sim(z_hat, z_ctx_i) / tau)-weighted
#      blend of context frames -> pixel output

# On the jepa-wms decoder (https://github.com/facebookresearch/jepa-wms)
# ------------------------------------------------------------------------
#   The available weights (vm2m_lpips_vj2vitgnorm_vitldec_dup_256_INet.pth.tar)
#   were trained with a ViT-G encoder (embed_dim=1408).  If you use the ViT-L
#   encoder here (embed_dim=1024) you need a linear projection 1024->1408
#   before feeding patch tokens into the decoder.  See the commented scaffold
#   at the bottom of this file.
# """

# import os
# from pathlib import Path
# from datetime import datetime

# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont

# import torch
# import torch.nn.functional as F

# # ─────────────────────────────────────────────────────────────────────────────
# #  CONFIG
# # ─────────────────────────────────────────────────────────────────────────────

# HF_REPO         = "facebook/vjepa2-vitl-fpc16-256-ssv2"
# DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# VIDEO_DIR       = "/project/3018078.02/MEG_ingmar/shorts/"
# VIDEO_PATH      = VIDEO_DIR + "bw_testclip_bouwval.mp4"

# CONTEXT_FRAMES  = 10
# TOTAL_FRAMES    = 15
# TEMPORAL_STRIDE = 1

# BLEND_TEMPERATURE   = 0.07
# AUTOREGRESSIVE_MODE = True

# RMSE_PLOT_MAX      = 0.20
# SSIM_ERR_PLOT_MAX  = 1.00
# BLUR_LOSS_PLOT_MAX = 1.00
# EDGE_ERR_PLOT_MAX  = 1.00
# FLOW_ERR_PLOT_MAX  = 0.20

# # ─────────────────────────────────────────────────────────────────────────────
# #  METRIC HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_ssim(gray_a, gray_b):
#     gray_a = gray_a.astype(np.float32)
#     gray_b = gray_b.astype(np.float32)
#     c1, c2 = (0.01 ** 2), (0.03 ** 2)
#     mu_a  = cv2.GaussianBlur(gray_a, (7, 7), 1.5)
#     mu_b  = cv2.GaussianBlur(gray_b, (7, 7), 1.5)
#     mu_a2, mu_b2, mu_ab = mu_a*mu_a, mu_b*mu_b, mu_a*mu_b
#     sig_a2 = cv2.GaussianBlur(gray_a*gray_a, (7,7), 1.5) - mu_a2
#     sig_b2 = cv2.GaussianBlur(gray_b*gray_b, (7,7), 1.5) - mu_b2
#     sig_ab = cv2.GaussianBlur(gray_a*gray_b, (7,7), 1.5) - mu_ab
#     num = (2*mu_ab+c1)*(2*sig_ab+c2)
#     den = (mu_a2+mu_b2+c1)*(sig_a2+sig_b2+c2)
#     return float(np.mean(num/(den+1e-8)))


# def compute_edge_f1(gray_a_u8, gray_b_u8):
#     ea = cv2.Canny(gray_a_u8, 100, 200) > 0
#     eb = cv2.Canny(gray_b_u8, 100, 200) > 0
#     tp = np.logical_and(ea,  eb ).sum()
#     fp = np.logical_and(~ea, eb ).sum()
#     fn = np.logical_and(ea,  ~eb).sum()
#     d  = 2*tp + fp + fn
#     return 1.0 if d == 0 else float(2*tp / d)


# def compute_optical_flow_error(gray_a_u8, gray_b_u8):
#     flow = cv2.calcOpticalFlowFarneback(
#         gray_a_u8, gray_b_u8, None,
#         pyr_scale=0.5, levels=3, winsize=15,
#         iterations=3, poly_n=5, poly_sigma=1.2, flags=0
#     )
#     mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     return float(min(np.mean(mag) / 50.0, 1.0))


# def compute_blur_loss(gt_u8, pred_u8):
#     gt_var   = float(cv2.Laplacian(gt_u8,   cv2.CV_32F).var())
#     pred_var = float(cv2.Laplacian(pred_u8, cv2.CV_32F).var())
#     return float(np.clip(1.0 - pred_var/(gt_var+1e-8), 0.0, 1.0))


# def compute_phase_stats(values, boundary_idx):
#     v = np.asarray(values, dtype=np.float32)
#     b = int(np.clip(boundary_idx, 0, v.size))
#     ctx, fut = v[:b], v[b:]
#     def ms(a):
#         return (float('nan'), float('nan')) if a.size == 0 else (float(np.mean(a)), float(np.std(a)))
#     cm, cs = ms(ctx)
#     fm, fs = ms(fut)
#     delta = float(fm - cm) if np.isfinite(cm) and np.isfinite(fm) else float('nan')
#     return dict(n_context=int(ctx.size), n_future=int(fut.size),
#                 context_mean=cm, context_std=cs,
#                 future_mean=fm, future_std=fs,
#                 delta_future_minus_context=delta)


# # ─────────────────────────────────────────────────────────────────────────────
# #  V-JEPA 2 ENCODER UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────

# def get_backbone(model):
#     """Extract the VJEPA2Model backbone from the HF classification wrapper."""
#     for attr in ("vjepa2", "model", "encoder", "backbone", "vision_model"):
#         if hasattr(model, attr):
#             sub = getattr(model, attr)
#             if hasattr(sub, "parameters"):
#                 return sub
#     named = [(n, type(m).__name__) for n, m in model.named_children()]
#     raise AttributeError(f"Cannot find ViT backbone. Named children: {named}")


# def preprocess_frame_for_vjepa(frame_bgr, processor, device, n_clip_frames=16):
#     """
#     Convert a single BGR uint8 frame to a V-JEPA2 input tensor.

#     Three things that must be correct:
#       1. Pass a (T, H, W, C) uint8 numpy array — NOT a list of PIL images.
#       2. Read "pixel_values_videos" from the processor output — NOT "pixel_values".
#       3. Result shape must be (1, T, C, H, W).
#     """
#     frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)      # (H, W, 3) uint8
#     clip_np   = np.stack([frame_rgb] * n_clip_frames, axis=0)   # (T, H, W, 3) uint8

#     inputs = processor(clip_np, return_tensors="pt")
#     pv     = inputs["pixel_values_videos"]                       # (1, T, C, H, W)

#     while pv.ndim < 5:
#         pv = pv.unsqueeze(0)
#     if pv.shape[0] != 1:
#         pv = pv.unsqueeze(0)

#     return pv.to(device)


# @torch.no_grad()
# def encode_frame(frame_bgr, backbone, processor, device, n_clip_frames=16):
#     """
#     Encode a single BGR frame. Returns (D,) L2-normalised latent on CPU.

#     VJEPA2Model.forward() takes pixel_values_videos=, not pixel_values=.
#     """
#     pv  = preprocess_frame_for_vjepa(frame_bgr, processor, device, n_clip_frames)
#     out = backbone(pixel_values_videos=pv)

#     hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
#     hidden = hidden.squeeze(0)          # (N_tokens, D)
#     z      = hidden.mean(dim=0)         # (D,)
#     z      = F.normalize(z, dim=0)
#     return z.cpu()


# # ─────────────────────────────────────────────────────────────────────────────
# #  LATENT-SPACE PREDICTION + PIXEL RECONSTRUCTION
# # ─────────────────────────────────────────────────────────────────────────────

# def predict_latent_linear(context_latents, steps_ahead=1):
#     """Linear velocity extrapolation in latent space, result L2-normalised."""
#     if len(context_latents) < 2:
#         return context_latents[-1].clone()
#     velocity = context_latents[-1] - context_latents[-2]
#     return F.normalize(context_latents[-1] + steps_ahead * velocity, dim=0)


# def reconstruct_pixels(predicted_latent, context_latents, context_frames_rgb, temperature=0.07):
#     """
#     Softmax-weighted blend of context frames using cosine similarity to
#     the predicted latent as blend weights.
#     Returns (H, W, 3) uint8 RGB.
#     """
#     sims    = torch.stack([torch.dot(predicted_latent, z) for z in context_latents])
#     weights = F.softmax(sims / temperature, dim=0).numpy()

#     h, w    = context_frames_rgb[0].shape[:2]
#     blended = np.zeros((h, w, 3), dtype=np.float32)
#     for wt, frame in zip(weights, context_frames_rgb):
#         blended += wt * frame.astype(np.float32)

#     return np.clip(blended, 0, 255).astype(np.uint8)


# # ─────────────────────────────────────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     from transformers import AutoVideoProcessor, AutoModelForVideoClassification

#     video_stem     = Path(VIDEO_PATH).stem
#     run_timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_output_dir = Path("predictions") / f"{video_stem}_vjepa2_{run_timestamp}"
#     run_output_dir.mkdir(parents=True, exist_ok=True)
#     print("Saving outputs to:", run_output_dir)

#     suffix       = "_autoregressive" if AUTOREGRESSIVE_MODE else ""
#     strategy_tag = "_fixfr"
#     mode_tag     = "_fwd"
#     stride_tag   = f"_s{TEMPORAL_STRIDE}"

#     cfg_path = run_output_dir / "run_config.txt"
#     with open(cfg_path, "w") as f:
#         f.write(f"run_timestamp: {run_timestamp}\n")
#         f.write(f"hf_repo: {HF_REPO}\n")
#         f.write(f"video_path: {VIDEO_PATH}\n")
#         f.write(f"device: {DEVICE}\n")
#         f.write(f"context_frames: {CONTEXT_FRAMES}\n")
#         f.write(f"total_frames: {TOTAL_FRAMES}\n")
#         f.write(f"temporal_stride: {TEMPORAL_STRIDE}\n")
#         f.write(f"blend_temperature: {BLEND_TEMPERATURE}\n")
#         f.write(f"autoregressive_mode: {AUTOREGRESSIVE_MODE}\n")
#         f.write("metrics: rmse, 1-ssim, blur_loss, 1-edge_f1, optical_flow_epe\n")
#     print("Saved run config to:", cfg_path)

#     # ── load model ─────────────────────────────────────────────────────────
#     print(f"\nLoading V-JEPA2 from {HF_REPO} ...")
#     model     = AutoModelForVideoClassification.from_pretrained(HF_REPO).to(DEVICE)
#     processor = AutoVideoProcessor.from_pretrained(HF_REPO)
#     model.eval()
#     print("Model loaded; device =", DEVICE)

#     backbone      = get_backbone(model)
#     n_clip_frames = 64 if "fpc64" in HF_REPO else 16
#     print(f"Backbone: {type(backbone).__name__}  |  clip frames: {n_clip_frames}")

#     # ── load video ─────────────────────────────────────────────────────────
#     print(f"\nReading video: {VIDEO_PATH}")
#     cap, raw_frames = cv2.VideoCapture(VIDEO_PATH), []
#     while len(raw_frames) < TOTAL_FRAMES:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         raw_frames.append(frame)
#         for _ in range(TEMPORAL_STRIDE - 1):
#             if not cap.grab():
#                 break
#     cap.release()

#     if len(raw_frames) < TOTAL_FRAMES:
#         raise ValueError(f"Need {TOTAL_FRAMES} frames, got {len(raw_frames)}.")
#     raw_frames = raw_frames[:TOTAL_FRAMES]
#     print(f"Loaded {len(raw_frames)} frames")

#     # Spatial size — processor handles normalisation internally, but we need
#     # consistently-sized frames for pixel metrics.
#     try:
#         img_size = processor.crop_size.get("height",
#                    processor.size.get("shortest_edge",
#                    processor.size.get("height", 256)))
#     except Exception:
#         img_size = 256

#     def to_rgb(bgr):
#         return cv2.cvtColor(cv2.resize(bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)

#     frames_rgb = [to_rgb(f) for f in raw_frames]

#     # ── encode context ─────────────────────────────────────────────────────
#     print(f"\nEncoding {CONTEXT_FRAMES} context frames ...")
#     context_latents, context_frames_rgb = [], []
#     for i in range(CONTEXT_FRAMES):
#         z = encode_frame(raw_frames[i], backbone, processor, DEVICE, n_clip_frames)
#         context_latents.append(z)
#         context_frames_rgb.append(frames_rgb[i])
#         print(f"  Frame {i+1}/{CONTEXT_FRAMES}  |z|={z.norm():.4f}  D={z.shape[0]}")

#     # ── predict future ─────────────────────────────────────────────────────
#     n_future             = TOTAL_FRAMES - CONTEXT_FRAMES
#     predicted_frames_rgb = list(context_frames_rgb)
#     running_latents      = list(context_latents)

#     print(f"\nPredicting {n_future} future frames ...")
#     for step in range(n_future):
#         z_pred     = predict_latent_linear(running_latents)
#         frame_pred = reconstruct_pixels(z_pred, running_latents, context_frames_rgb,
#                                         BLEND_TEMPERATURE)
#         predicted_frames_rgb.append(frame_pred)
#         if AUTOREGRESSIVE_MODE:
#             running_latents.append(F.normalize(z_pred, dim=0))
#         print(f"  Future frame {step+1}/{n_future}")

#     # ── save PNGs ──────────────────────────────────────────────────────────
#     for i, fr in enumerate(predicted_frames_rgb):
#         png_path = run_output_dir / f"pred_{i+1:02d}{suffix}{strategy_tag}{mode_tag}{stride_tag}.png"
#         cv2.imwrite(str(png_path), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
#     print(f"Saved {TOTAL_FRAMES} PNGs")

#     # ── prediction GIF ─────────────────────────────────────────────────────
#     gif_path = run_output_dir / f"predictions{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
#     pil_pred = [Image.fromarray(f) for f in predicted_frames_rgb[1:]]
#     pil_pred[0].save(str(gif_path), save_all=True, append_images=pil_pred[1:],
#                      duration=150, loop=0, optimize=False)
#     print("SAVED predictions GIF  ->", gif_path)

#     # ── ground-truth GIF ───────────────────────────────────────────────────
#     gt_gif = run_output_dir / f"groundtruth{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
#     gt_pil = [Image.fromarray(f) for f in frames_rgb[1:]]
#     gt_pil[0].save(str(gt_gif), save_all=True, append_images=gt_pil[1:],
#                    duration=150, loop=0, optimize=False)
#     print("SAVED ground-truth GIF ->", gt_gif)

#     # ── framewise metrics ──────────────────────────────────────────────────
#     print("\nComputing metrics ...")
#     rmse_v, ssim_v, blur_v, edge_v, flow_v = [], [], [], [], []

#     for idx, (gt_rgb, pr_rgb) in enumerate(zip(frames_rgb, predicted_frames_rgb)):
#         gt_f  = gt_rgb.astype(np.float32)  / 255.0
#         pr_f  = pr_rgb.astype(np.float32) / 255.0
#         rmse_v.append(float(np.sqrt(np.mean((gt_f - pr_f)**2))))

#         gt_g  = cv2.cvtColor(gt_rgb,  cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#         pr_g  = cv2.cvtColor(pr_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#         ssim_v.append(float(1.0 - compute_ssim(gt_g, pr_g)))

#         gt_u8  = (gt_g  * 255).astype(np.uint8)
#         pr_u8  = (pr_g * 255).astype(np.uint8)
#         blur_v.append(compute_blur_loss(gt_u8, pr_u8))
#         edge_v.append(float(1.0 - compute_edge_f1(gt_u8, pr_u8)))

#         if idx < len(frames_rgb) - 1:
#             gn = cv2.cvtColor(frames_rgb[idx+1],           cv2.COLOR_RGB2GRAY).astype(np.uint8)
#             pn = cv2.cvtColor(predicted_frames_rgb[idx+1], cv2.COLOR_RGB2GRAY).astype(np.uint8)
#             flow_v.append(float(abs(compute_optical_flow_error(gt_u8, gn) -
#                                     compute_optical_flow_error(pr_u8, pn))))
#         else:
#             flow_v.append(flow_v[-1] if flow_v else 0.0)

#     flow_v = [0.0] + flow_v[:-1]   # same shift as PredNet script

#     # ── metrics summary ────────────────────────────────────────────────────
#     phase_b = CONTEXT_FRAMES if AUTOREGRESSIVE_MODE else len(rmse_v)
#     series  = [("RMSE", rmse_v), ("1-SSIM", ssim_v), ("Blur Loss", blur_v),
#                ("1-EdgeF1", edge_v), ("Flow EPE", flow_v)]

#     summary_path = run_output_dir / f"metrics_summary{suffix}{strategy_tag}{mode_tag}{stride_tag}.txt"
#     lines = [
#         f"model: {HF_REPO}",
#         "prediction_strategy: velocity_extrapolation + cosine_blend",
#         f"mode: {'autoregressive' if AUTOREGRESSIVE_MODE else 'standard'}",
#         f"boundary_index: {phase_b}",
#         "",
#         "metric       ctx_mean  ctx_std  fut_mean  fut_std  delta(fut-ctx)  n_ctx  n_fut",
#         "--------------------------------------------------------------------------------",
#     ]
#     for name, vals in series:
#         s = compute_phase_stats(vals, phase_b)
#         lines.append(f"{name:11s} "
#                      f"{s['context_mean']:8.4f} {s['context_std']:8.4f} "
#                      f"{s['future_mean']:8.4f} {s['future_std']:8.4f} "
#                      f"{s['delta_future_minus_context']:14.4f} "
#                      f"{s['n_context']:6d} {s['n_future']:6d}")

#     with open(summary_path, "w") as f:
#         f.write("\n".join(lines) + "\n")
#     print("Saved metrics summary to:", summary_path)
#     print("\n".join(lines))

#     # ── comparison GIF ─────────────────────────────────────────────────────
#     print("\nBuilding comparison GIF ...")
#     try:
#         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
#     except Exception:
#         try:
#             font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
#         except Exception:
#             font = ImageFont.load_default()

#     H_f, W_f      = frames_rgb[0].shape[:2]
#     title_h       = 20
#     plot_h        = 285
#     pad_l, pad_r  = 36, 12
#     pad_t, pad_b  = 12, 22

#     plot_metrics = [
#         ("RMSE",      rmse_v, RMSE_PLOT_MAX,      ( 50, 180, 255)),
#         ("1-SSIM",    ssim_v, SSIM_ERR_PLOT_MAX,  (140, 230, 160)),
#         ("Blur Loss", blur_v, BLUR_LOSS_PLOT_MAX,  (255, 170,  90)),
#         ("1-EdgeF1",  edge_v, EDGE_ERR_PLOT_MAX,  (250, 200, 120)),
#         ("Flow EPE",  flow_v, FLOW_ERR_PLOT_MAX,  (200, 150, 255)),
#     ]

#     cmp_frames = []
#     for fi, (gt_f, pr_f) in enumerate(zip(frames_rgb[1:], predicted_frames_rgb[1:]), start=1):
#         cw = W_f * 2
#         ch = title_h + H_f + plot_h
#         cv_img = Image.new("RGB", (cw, ch), (0, 0, 0))
#         draw   = ImageDraw.Draw(cv_img)

#         if AUTOREGRESSIVE_MODE and fi < CONTEXT_FRAMES:
#             g_lbl, p_lbl = "Context (GT)", "Context (Pred)"
#         elif AUTOREGRESSIVE_MODE:
#             g_lbl, p_lbl = "Future (GT)", "Future (Pred)"
#         else:
#             g_lbl, p_lbl = "Ground Truth", "Prediction"

#         draw.text((W_f//2 - 40, 2),       g_lbl, fill=(255,255,255), font=font)
#         draw.text((W_f + W_f//2 - 30, 2), p_lbl, fill=(255,255,255), font=font)
#         cv_img.paste(Image.fromarray(gt_f), (0,   title_h))
#         cv_img.paste(Image.fromarray(pr_f), (W_f, title_h))

#         px0 = pad_l;       py0 = title_h + H_f + pad_t
#         px1 = cw - pad_r;  py1 = ch - pad_b
#         n_m = len(plot_metrics);  row_gap = 5
#         row_h = (py1 - py0 - (n_m-1)*row_gap) // n_m

#         for mi, (mn, mv, mm, lc) in enumerate(plot_metrics):
#             ry0 = py0 + mi * (row_h + row_gap)
#             ry1 = ry0 + row_h
#             draw.line([(px0, ry0), (px0, ry1)], fill=(180,180,180), width=1)
#             draw.line([(px0, ry1), (px1, ry1)], fill=(180,180,180), width=1)
#             draw.text((40, ry0), mn, fill=(220,220,220), font=font)
#             draw.text((px0-25, ry0), f"{mm:.1f}", fill=(150,150,150), font=font)

#             if AUTOREGRESSIVE_MODE and 0 <= CONTEXT_FRAMES-1 < len(mv) and len(mv) > 1:
#                 sx = int(px0 + (CONTEXT_FRAMES-1)/(len(mv)-1)*(px1-px0))
#                 draw.line([(sx, ry0), (sx, ry1)], fill=(80,80,80), width=1)

#             if len(mv) == 1:
#                 pts = [((px0+px1)//2, ry1)]
#             else:
#                 pts = [(int(px0 + k/(len(mv)-1)*(px1-px0)),
#                         int(ry1 - (v/mm)*(ry1-ry0))) for k, v in enumerate(mv)]

#             if len(pts) > 1:
#                 draw.line(pts, fill=lc, width=2)

#             ck = max(0, min(fi, len(mv)-1))
#             cx, cy = pts[ck]
#             draw.ellipse((cx-3, cy-3, cx+3, cy+3), fill=(255,60,60))
#             draw.text((cx-40, max(ry0, cy-15)), f"{mv[ck]:.3f}", fill=(255,120,120), font=font)

#         draw.text((px0,    py1+4), "0",                    fill=(180,180,180), font=font)
#         draw.text((px1-22, py1+4), str(len(rmse_v)-1),    fill=(180,180,180), font=font)
#         cmp_frames.append(cv_img)

#     cmp_gif = run_output_dir / f"comparison{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
#     cmp_frames[0].save(str(cmp_gif), save_all=True, append_images=cmp_frames[1:],
#                        duration=150, loop=0, optimize=False)
#     print("SAVED comparison GIF   ->", cmp_gif)
#     print("\nAll done. Output directory:", run_output_dir)


# if __name__ == "__main__":
#     main()
