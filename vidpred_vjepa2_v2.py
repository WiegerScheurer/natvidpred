#!/usr/bin/env python3
"""
vidpred_vjepa2_vitg.py  —  V-JEPA2 ViT-G video prediction
See docstring below for full explanation of design choices.
"""

import os, sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

HF_REPO            = "facebook/vjepa2-vitg-fpc64-256"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

JEPA_WMS_REPO_DIR  = "/project/3018078.02/natvidpred_workspace/jepa-wms"
DECODER_CKPT_PATH  = "/project/3018078.02/natvidpred_workspace/vm2m_lpips_vj2vitgnorm_vitldec_dup_256_INet.pth.tar"

VIDEO_DIR          = "/project/3018078.02/MEG_ingmar/shorts/"
VIDEO_PATH         = VIDEO_DIR + "bw_testclip_bouwval.mp4"

CONTEXT_FRAMES     = 10
TOTAL_FRAMES       = 15
TEMPORAL_STRIDE    = 1

IMG_SIZE           = 256
PATCH_SIZE         = 16
N_PATCHES          = (IMG_SIZE // PATCH_SIZE) ** 2   # 256
N_CLIP_FRAMES      = 64
TUBELET_DEPTH      = 2

AUTOREGRESSIVE_MODE = True

RMSE_PLOT_MAX       = 0.20
SSIM_ERR_PLOT_MAX   = 1.00
BLUR_LOSS_PLOT_MAX  = 1.00
EDGE_ERR_PLOT_MAX   = 1.00
FLOW_ERR_PLOT_MAX   = 0.20

# ─────────────────────────────────────────────────────────────────────────────
#  METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(gray_a, gray_b):
    a,b = gray_a.astype(np.float32), gray_b.astype(np.float32)
    c1,c2 = 0.01**2, 0.03**2
    ma,mb = cv2.GaussianBlur(a,(7,7),1.5), cv2.GaussianBlur(b,(7,7),1.5)
    ma2,mb2,mab = ma*ma,mb*mb,ma*mb
    sa2=cv2.GaussianBlur(a*a,(7,7),1.5)-ma2
    sb2=cv2.GaussianBlur(b*b,(7,7),1.5)-mb2
    sab=cv2.GaussianBlur(a*b,(7,7),1.5)-mab
    return float(np.mean(((2*mab+c1)*(2*sab+c2))/((ma2+mb2+c1)*(sa2+sb2+c2)+1e-8)))

def compute_edge_f1(a_u8, b_u8):
    ea=cv2.Canny(a_u8,100,200)>0; eb=cv2.Canny(b_u8,100,200)>0
    tp=np.logical_and(ea,eb).sum(); fp=np.logical_and(~ea,eb).sum()
    fn=np.logical_and(ea,~eb).sum(); d=2*tp+fp+fn
    return 1.0 if d==0 else float(2*tp/d)

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
#  OPTICAL FLOW WARP  —  pixel-space fallback, cannot diverge
# ─────────────────────────────────────────────────────────────────────────────
# Why the OLD per-patch cosine blend failed:
#   Linear token extrapolation sends vectors off the representation manifold
#   after step 2.  Cosine similarities with far-off-manifold tokens collapse
#   to near-zero / near-uniform → softmax is flat → output = mean of all
#   context frames = grey blob, darkening each step.
#
# Optical flow warp operates entirely in pixel space.  A constant-velocity
# motion model is estimated from the last two context frames and applied
# cumulatively.  It can never produce dark blocks.

def compute_dense_flow(frame_a_rgb, frame_b_rgb):
    """Dense Farneback optical flow from a→b. Returns (H,W,2) in pixels/frame."""
    ga = cv2.cvtColor(frame_a_rgb, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(frame_b_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        ga, gb, None,
        pyr_scale=0.5, levels=5, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0
    )

def warp_frame_flow(frame_rgb, flow):
    """Warp RGB frame forward by flow (H,W,2). Uses BORDER_REPLICATE at edges."""
    H,W = frame_rgb.shape[:2]
    gx,gy = np.meshgrid(np.arange(W,dtype=np.float32),
                        np.arange(H,dtype=np.float32))
    map_x = (gx + flow[...,0]).astype(np.float32)
    map_y = (gy + flow[...,1]).astype(np.float32)
    return cv2.remap(frame_rgb, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

def predict_frames_flow_warp(context_frames_rgb):
    """
    Predict all future frames using constant-velocity optical flow extrapolation.
    Returns list of n_future (H,W,3) uint8 RGB frames.
    """
    n_future     = TOTAL_FRAMES - CONTEXT_FRAMES
    last_frame   = context_frames_rgb[-1]
    prev_frame   = context_frames_rgb[-2]
    vel_flow     = compute_dense_flow(prev_frame, last_frame)
    mean_mag     = float(np.mean(np.sqrt(vel_flow[...,0]**2+vel_flow[...,1]**2)))
    print(f"  Flow velocity: mean magnitude = {mean_mag:.3f} px/frame")

    predicted = []
    for step in range(1, n_future+1):
        warped = warp_frame_flow(last_frame, (step * vel_flow).astype(np.float32))
        predicted.append(warped)
        print(f"  Flow-warp future frame {step}/{n_future}")
    return predicted

# ─────────────────────────────────────────────────────────────────────────────
#  DECODER LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _inspect_and_build_decoder(state_dict, device):
    """
    Read checkpoint key shapes → infer architecture → build custom ViTDecoder.
    Returns decoder module or None if inference fails.
    """
    # Strip 'module.' prefix (from DataParallel checkpoints)
    sd_cleaned = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        sd_cleaned[key] = v
    
    # Infer architecture from checkpoint
    embed_dim = None
    in_dim = None
    n_blocks = 0
    num_heads = 16
    
    # Get embed_dim from decoder_embed.weight
    if "decoder_embed.weight" in sd_cleaned:
        in_dim = sd_cleaned["decoder_embed.weight"].shape[1]
        embed_dim = sd_cleaned["decoder_embed.weight"].shape[0]
    
    # Count decoder blocks
    for k in sd_cleaned:
        if "decoder_blocks." in k:
            try:
                block_idx = int(k.split("decoder_blocks.")[1].split(".")[0])
                n_blocks = max(n_blocks, block_idx + 1)
            except:
                pass
    
    # Calculate num_heads from qkv weight
    for k, v in sd_cleaned.items():
        if "decoder_blocks.0.attn.qkv.weight" in k and v.ndim == 2:
            qkv_out = v.shape[0]
            num_heads = qkv_out // (3 * embed_dim)
            break
    
    print(f"  [arch] embed_dim={embed_dim}  in_dim={in_dim}  n_blocks={n_blocks}  num_heads={num_heads}")
    if embed_dim is None or in_dim is None:
        print("  [arch] Cannot infer architecture from checkpoint keys")
        return None

    class CustomAttention(nn.Module):
        """Custom attention matching checkpoint format."""
        def __init__(self, dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
        
        def forward(self, x):
            B, N, D = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, D)
            x = self.proj(x)
            return x

    class CustomBlock(nn.Module):
        """Custom transformer block matching checkpoint format."""
        def __init__(self, dim, num_heads, mlp_dim):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = CustomAttention(dim, num_heads)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, dim)
            )
        
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class ViTDecoderCustom(nn.Module):
        """Custom decoder matching checkpoint structure."""
        def __init__(self):
            super().__init__()
            self.decoder_embed = nn.Linear(in_dim, embed_dim)
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
            
            self.decoder_blocks = nn.ModuleList([
                CustomBlock(embed_dim, num_heads, embed_dim * 4)
                for _ in range(n_blocks)
            ])
            
            self.norm = nn.LayerNorm(embed_dim)
            self.pred_head = nn.Linear(embed_dim, PATCH_SIZE * PATCH_SIZE * 3)

        def forward(self, x):
            B, N, _ = x.shape
            x = self.decoder_embed(x)
            x = x + self.decoder_pos_embed[:, :N, :]
            
            for block in self.decoder_blocks:
                x = block(x)
            
            x = self.norm(x)
            x = self.pred_head(x)
            
            G = int(N ** 0.5)
            P = PATCH_SIZE
            x = x.reshape(B, G, G, P, P, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, G * P, G * P)
            return torch.sigmoid(x)

    dec = ViTDecoderCustom()
    missing, unexpected = dec.load_state_dict(sd_cleaned, strict=False)
    ratio = (len(sd_cleaned) - len(unexpected)) / max(len(sd_cleaned), 1)
    print(f"  [weights] load_ratio={ratio:.2f}  missing={len(missing)}  unexpected={len(unexpected)}")
    if ratio < 0.5:
        print("  [weights] <50% loaded — architecture mismatch, skipping")
        return None
    dec.eval().to(device)
    print("  [weights] ViTDecoder ready")
    return dec


def try_load_decoder(ckpt_path, repo_dir, device):
    if not Path(ckpt_path).exists():
        print(f"[decoder] Not found: {ckpt_path}"); return None

    print(f"[decoder] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state dict
    sd = None
    for key in ("decoder","model","state_dict"):
        if isinstance(ckpt,dict) and key in ckpt:
            c = ckpt[key]
            if isinstance(c,dict) and any(hasattr(v,'shape') for v in c.values()):
                sd = c; print(f"[decoder] State dict under '{key}'  ({len(sd)} tensors)"); break
    if sd is None and isinstance(ckpt,dict) and any(hasattr(v,'shape') for v in ckpt.values()):
        sd = ckpt; print(f"[decoder] Top-level as state dict  ({len(sd)} tensors)")
    if sd is None:
        print("[decoder] Cannot extract state dict"); return None

    # Print first 15 keys so user can see the architecture
    print("  Sample keys:")
    for k,v in list(sd.items())[:15]:
        print(f"    {k:60s}  {tuple(v.shape)}")

    # Stage 1: try jepa-wms imports
    if Path(repo_dir).exists():
        for sub in ["","src",os.path.join("src","models"),"models"]:
            p = str(Path(repo_dir)/sub)
            if p not in sys.path: sys.path.insert(0,p)
        for mod_name,cls_names in [
            ("models.decoder",["VideoDecoder","ViTDecoder","Decoder","VisionTransformerDecoder"]),
            ("decoder",       ["VideoDecoder","ViTDecoder","Decoder","VisionTransformerDecoder"]),
            ("models.vit",    ["VisionTransformerDecoder","ViTDecoder"]),
        ]:
            try:
                mod = __import__(mod_name, fromlist=cls_names)
                for cn in cls_names:
                    if not hasattr(mod,cn): continue
                    cls = getattr(mod,cn)
                    for kw in [
                        dict(embed_dim=1024,depth=8,num_heads=16,patch_size=PATCH_SIZE,
                             img_size=IMG_SIZE,num_patches=N_PATCHES),
                        dict(embed_dim=1024,depth=8,num_heads=16),
                        dict()
                    ]:
                        try:
                            dec = cls(**kw)
                            miss,_ = dec.load_state_dict(sd,strict=False)
                            dec.eval().to(device)
                            print(f"[decoder] {cn} loaded (missing={len(miss)})")
                            return dec
                        except Exception: continue
            except ImportError: continue

    # Stage 2: build from key inspection
    print("[decoder] Trying MAEDecoder from checkpoint introspection ...")
    return _inspect_and_build_decoder(sd, device)


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODER
# ─────────────────────────────────────────────────────────────────────────────

def get_backbone(model):
    for attr in ("vjepa2","model","encoder","backbone","vision_model"):
        if hasattr(model,attr):
            sub=getattr(model,attr)
            if hasattr(sub,"parameters"): return sub
    raise AttributeError(f"No backbone found. Children: {[n for n,_ in model.named_children()]}")

def frame_to_clip_tensor(frame_bgr, processor, device):
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    clip = np.stack([rgb]*N_CLIP_FRAMES, axis=0)
    pv   = processor(clip, return_tensors="pt")["pixel_values_videos"]
    while pv.ndim<5: pv=pv.unsqueeze(0)
    return pv.to(device)

@torch.no_grad()
def encode_frame_full_tokens(frame_bgr, backbone, processor, device):
    """Returns (N_patches=256, D=1408) from middle temporal slice."""
    pv  = frame_to_clip_tensor(frame_bgr, processor, device)
    out = backbone(pixel_values_videos=pv)
    hidden = (out.last_hidden_state if hasattr(out,"last_hidden_state") else out[0]).squeeze(0)
    n_t = N_CLIP_FRAMES//TUBELET_DEPTH; expected=n_t*N_PATCHES
    if hidden.shape[0]==expected+1: hidden=hidden[1:]
    if hidden.shape[0]==expected:
        mid=n_t//2; hidden=hidden[mid*N_PATCHES:(mid+1)*N_PATCHES]
    else:
        hidden=hidden[:N_PATCHES]
        if hidden.shape[0]<N_PATCHES:
            pad=N_PATCHES-hidden.shape[0]
            hidden=torch.cat([hidden,hidden[-1:].expand(pad,-1)],dim=0)
    return hidden.cpu()

def predict_tokens_linear(token_list, steps_ahead=1):
    if len(token_list)<2: return token_list[-1].clone()
    return token_list[-1] + steps_ahead*(token_list[-1]-token_list[-2])

@torch.no_grad()
def decode_tokens(pred_tokens, decoder, device):
    t   = pred_tokens.unsqueeze(0).to(device)
    out = decoder(t)
    if out.ndim==4:
        img=out[0].permute(1,2,0).cpu().numpy()
    elif out.ndim==3:
        G=int(out.shape[1]**0.5); P=PATCH_SIZE
        img=out[0].reshape(G,G,P,P,3).permute(4,0,2,1,3).reshape(3,G*P,G*P).permute(1,2,0).cpu().numpy()
    else:
        raise ValueError(f"Bad decoder output shape: {out.shape}")
    img=np.clip(img,0.0,1.0) if img.max()<=1.1 else np.clip(img/255.0,0.0,1.0)
    return (img*255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoVideoProcessor, AutoModelForVideoClassification

    video_stem     = Path(VIDEO_PATH).stem
    run_timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path("predictions")/f"{video_stem}_vjepa2g_{run_timestamp}"
    run_output_dir.mkdir(parents=True,exist_ok=True)
    print("Saving outputs to:", run_output_dir)

    suffix=("_autoregressive" if AUTOREGRESSIVE_MODE else "")
    strategy_tag="_fixfr"; mode_tag="_fwd"; stride_tag=f"_s{TEMPORAL_STRIDE}"

    with open(run_output_dir/"run_config.txt","w") as f:
        for k,v in [("hf_repo",HF_REPO),("video_path",VIDEO_PATH),("device",DEVICE),
                    ("context_frames",CONTEXT_FRAMES),("total_frames",TOTAL_FRAMES),
                    ("temporal_stride",TEMPORAL_STRIDE),("autoregressive_mode",AUTOREGRESSIVE_MODE),
                    ("decoder_ckpt",DECODER_CKPT_PATH)]:
            f.write(f"{k}: {v}\n")

    # ── decoder ────────────────────────────────────────────────────────────
    print("\n── Loading jepa-wms decoder ──────────────────────────────────────")
    decoder       = try_load_decoder(DECODER_CKPT_PATH, JEPA_WMS_REPO_DIR, DEVICE)
    using_decoder = decoder is not None
    print(f"\nDecoder mode: {'jepa-wms ViT-L decoder' if using_decoder else 'optical flow warp (fallback)'}")

    # ── encoder ────────────────────────────────────────────────────────────
    print(f"\n── Loading encoder: {HF_REPO} ───────────────────────────────────")
    model     = AutoModelForVideoClassification.from_pretrained(HF_REPO).to(DEVICE)
    processor = AutoVideoProcessor.from_pretrained(HF_REPO)
    model.eval()
    backbone  = get_backbone(model)
    print(f"Encoder: {type(backbone).__name__}  device={DEVICE}  embed_dim={getattr(backbone.config,'hidden_size','?')}")

    # ── video ──────────────────────────────────────────────────────────────
    print(f"\nReading video: {VIDEO_PATH}")
    cap,raw_frames = cv2.VideoCapture(VIDEO_PATH),[]
    while len(raw_frames)<TOTAL_FRAMES:
        ret,frame=cap.read()
        if not ret: break
        raw_frames.append(frame)
        for _ in range(TEMPORAL_STRIDE-1):
            if not cap.grab(): break
    cap.release()
    if len(raw_frames)<TOTAL_FRAMES:
        raise ValueError(f"Need {TOTAL_FRAMES} frames, got {len(raw_frames)}.")
    raw_frames=raw_frames[:TOTAL_FRAMES]
    print(f"Loaded {len(raw_frames)} frames")

    def to_rgb(bgr): return cv2.cvtColor(cv2.resize(bgr,(IMG_SIZE,IMG_SIZE)),cv2.COLOR_BGR2RGB)
    frames_rgb=[to_rgb(f) for f in raw_frames]

    # ── encode context ─────────────────────────────────────────────────────
    print(f"\nEncoding {CONTEXT_FRAMES} context frames ...")
    context_tokens=[]; context_frames_rgb=[]
    for i in range(CONTEXT_FRAMES):
        if using_decoder:
            tok=encode_frame_full_tokens(raw_frames[i],backbone,processor,DEVICE)
            context_tokens.append(tok)
        context_frames_rgb.append(frames_rgb[i])
        print(f"  Frame {i+1}/{CONTEXT_FRAMES}"+(f"  tokens={tuple(context_tokens[-1].shape)}" if using_decoder else ""))

    # ── predict ────────────────────────────────────────────────────────────
    n_future=TOTAL_FRAMES-CONTEXT_FRAMES
    predicted_frames_rgb=list(context_frames_rgb)
    print(f"\nPredicting {n_future} future frames ...")

    if using_decoder:
        running_tokens=list(context_tokens)
        for step in range(n_future):
            pred_tok=predict_tokens_linear(running_tokens,steps_ahead=1)
            try:
                frame_pred=decode_tokens(pred_tok,decoder,DEVICE); mode_str="decoder"
            except Exception as e:
                print(f"  [step {step+1}] decoder error: {e}, using flow-warp")
                fp=predict_frames_flow_warp(context_frames_rgb); frame_pred=fp[step]; mode_str="flow-warp"
            predicted_frames_rgb.append(frame_pred)
            if AUTOREGRESSIVE_MODE: running_tokens.append(pred_tok)
            print(f"  Future frame {step+1}/{n_future}  [{mode_str}]")
    else:
        print("  Computing optical flow ...")
        for fp in predict_frames_flow_warp(context_frames_rgb):
            predicted_frames_rgb.append(fp)

    # ── PNGs ───────────────────────────────────────────────────────────────
    for i,fr in enumerate(predicted_frames_rgb):
        cv2.imwrite(str(run_output_dir/f"pred_{i+1:02d}{suffix}{strategy_tag}{mode_tag}{stride_tag}.png"),
                    cv2.cvtColor(fr,cv2.COLOR_RGB2BGR))
    print(f"Saved {TOTAL_FRAMES} PNGs")

    # ── GIFs ───────────────────────────────────────────────────────────────
    def save_gif(frames, path):
        pil=[Image.fromarray(f) for f in frames]
        pil[0].save(str(path),save_all=True,append_images=pil[1:],duration=150,loop=0,optimize=False)

    gif_path=run_output_dir/f"predictions{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
    save_gif(predicted_frames_rgb[1:], gif_path); print("SAVED predictions GIF  ->", gif_path)

    gt_gif=run_output_dir/f"groundtruth{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
    save_gif(frames_rgb[1:], gt_gif); print("SAVED ground-truth GIF ->", gt_gif)

    # ── metrics ────────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    rmse_v,ssim_v,blur_v,edge_v,flow_v=[],[],[],[],[]
    for idx,(gt_rgb,pr_rgb) in enumerate(zip(frames_rgb,predicted_frames_rgb)):
        gt_f=gt_rgb.astype(np.float32)/255.0; pr_f=pr_rgb.astype(np.float32)/255.0
        rmse_v.append(float(np.sqrt(np.mean((gt_f-pr_f)**2))))
        gt_g=cv2.cvtColor(gt_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        pr_g=cv2.cvtColor(pr_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        ssim_v.append(float(1.0-compute_ssim(gt_g,pr_g)))
        gt_u8=(gt_g*255).astype(np.uint8); pr_u8=(pr_g*255).astype(np.uint8)
        blur_v.append(compute_blur_loss(gt_u8,pr_u8))
        edge_v.append(float(1.0-compute_edge_f1(gt_u8,pr_u8)))
        if idx<len(frames_rgb)-1:
            gn=cv2.cvtColor(frames_rgb[idx+1],cv2.COLOR_RGB2GRAY).astype(np.uint8)
            pn=cv2.cvtColor(predicted_frames_rgb[idx+1],cv2.COLOR_RGB2GRAY).astype(np.uint8)
            flow_v.append(float(abs(compute_optical_flow_error(gt_u8,gn)-compute_optical_flow_error(pr_u8,pn))))
        else:
            flow_v.append(flow_v[-1] if flow_v else 0.0)
    flow_v=[0.0]+flow_v[:-1]

    phase_b=CONTEXT_FRAMES if AUTOREGRESSIVE_MODE else len(rmse_v)
    series=[("RMSE",rmse_v),("1-SSIM",ssim_v),("Blur Loss",blur_v),("1-EdgeF1",edge_v),("Flow EPE",flow_v)]
    summary_path=run_output_dir/f"metrics_summary{suffix}{strategy_tag}{mode_tag}{stride_tag}.txt"
    lines=[f"model: {HF_REPO}",
           f"decoder: {'jepa-wms ViT-L' if using_decoder else 'optical-flow-warp'}",
           f"boundary_index: {phase_b}","",
           "metric       ctx_mean  ctx_std  fut_mean  fut_std  delta(fut-ctx)  n_ctx  n_fut",
           "--------------------------------------------------------------------------------"]
    for name,vals in series:
        s=compute_phase_stats(vals,phase_b)
        lines.append(f"{name:11s} {s['context_mean']:8.4f} {s['context_std']:8.4f} "
                     f"{s['future_mean']:8.4f} {s['future_std']:8.4f} "
                     f"{s['delta_future_minus_context']:14.4f} {s['n_context']:6d} {s['n_future']:6d}")
    with open(summary_path,"w") as f: f.write("\n".join(lines)+"\n")
    print("Saved metrics summary to:", summary_path)
    print("\n".join(lines))

    # ── comparison GIF ─────────────────────────────────────────────────────
    print("\nBuilding comparison GIF ...")
    try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",10)
    except Exception:
        try: font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",10)
        except: font=ImageFont.load_default()

    H_f,W_f=frames_rgb[0].shape[:2]
    title_h=20; plot_h=285; pad_l=36; pad_r=12; pad_t=12; pad_b=22
    plot_metrics=[("RMSE",rmse_v,RMSE_PLOT_MAX,(50,180,255)),("1-SSIM",ssim_v,SSIM_ERR_PLOT_MAX,(140,230,160)),
                  ("Blur Loss",blur_v,BLUR_LOSS_PLOT_MAX,(255,170,90)),("1-EdgeF1",edge_v,EDGE_ERR_PLOT_MAX,(250,200,120)),
                  ("Flow EPE",flow_v,FLOW_ERR_PLOT_MAX,(200,150,255))]
    cmp_frames=[]
    for fi,(gt_f,pr_f) in enumerate(zip(frames_rgb[1:],predicted_frames_rgb[1:]),start=1):
        cw=W_f*2; ch=title_h+H_f+plot_h
        ci=Image.new("RGB",(cw,ch),(0,0,0)); draw=ImageDraw.Draw(ci)
        g_lbl=("Context (GT)" if fi<CONTEXT_FRAMES else "Future (GT)") if AUTOREGRESSIVE_MODE else "Ground Truth"
        p_lbl=("Context (Pred)" if fi<CONTEXT_FRAMES else "Future (Pred)") if AUTOREGRESSIVE_MODE else "Prediction"
        draw.text((W_f//2-40,2),g_lbl,fill=(255,255,255),font=font)
        draw.text((W_f+W_f//2-30,2),p_lbl,fill=(255,255,255),font=font)
        ci.paste(Image.fromarray(gt_f),(0,title_h)); ci.paste(Image.fromarray(pr_f),(W_f,title_h))
        px0=pad_l; py0=title_h+H_f+pad_t; px1=cw-pad_r; py1=ch-pad_b
        n_m=len(plot_metrics); row_gap=5; row_h=(py1-py0-(n_m-1)*row_gap)//n_m
        for mi,(mn,mv,mm,lc) in enumerate(plot_metrics):
            ry0=py0+mi*(row_h+row_gap); ry1=ry0+row_h
            draw.line([(px0,ry0),(px0,ry1)],fill=(180,180,180),width=1)
            draw.line([(px0,ry1),(px1,ry1)],fill=(180,180,180),width=1)
            draw.text((40,ry0),mn,fill=(220,220,220),font=font)
            draw.text((px0-25,ry0),f"{mm:.1f}",fill=(150,150,150),font=font)
            if AUTOREGRESSIVE_MODE and 0<=CONTEXT_FRAMES-1<len(mv) and len(mv)>1:
                sx=int(px0+(CONTEXT_FRAMES-1)/(len(mv)-1)*(px1-px0))
                draw.line([(sx,ry0),(sx,ry1)],fill=(80,80,80),width=1)
            pts=[(int(px0+k/(len(mv)-1)*(px1-px0)),int(ry1-(v/mm)*(ry1-ry0)))
                 for k,v in enumerate(mv)] if len(mv)>1 else [((px0+px1)//2,ry1)]
            if len(pts)>1: draw.line(pts,fill=lc,width=2)
            ck=max(0,min(fi,len(mv)-1)); cx,cy=pts[ck]
            draw.ellipse((cx-3,cy-3,cx+3,cy+3),fill=(255,60,60))
            draw.text((cx-40,max(ry0,cy-15)),f"{mv[ck]:.3f}",fill=(255,120,120),font=font)
        draw.text((px0,py1+4),"0",fill=(180,180,180),font=font)
        draw.text((px1-22,py1+4),str(len(rmse_v)-1),fill=(180,180,180),font=font)
        cmp_frames.append(ci)

    cmp_gif=run_output_dir/f"comparison{suffix}{strategy_tag}{mode_tag}{stride_tag}.gif"
    cmp_frames[0].save(str(cmp_gif),save_all=True,append_images=cmp_frames[1:],duration=150,loop=0,optimize=False)
    print("SAVED comparison GIF   ->", cmp_gif)
    print("\nAll done. Output directory:", run_output_dir)

if __name__=="__main__":
    main()
