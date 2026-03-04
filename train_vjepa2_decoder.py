"""
Train a pixel decoder for V-JEPA 2 patch tokens
=================================================
Maps V-JEPA 2 encoder patch tokens  [B, 576, 1408]  →  pixel frames  [B, 3, 384, 384].

CRITICAL ENCODING NOTE
-----------------------
V-JEPA 2 uses *full temporal attention* across the entire 64-frame clip.
Every spatial token at every temporal position is contextualised by ALL other
tokens in the clip.  This means:
  - A frame encoded alone  ≠  the same frame encoded as part of a 64-frame clip.
  - The decoder MUST be trained on tokens extracted from full-clip encodings,
    because at inference time the predicted tokens come from full-clip encodings.
  - The old approach (encode each frame individually, duplicated 2×) was wrong.

This script encodes full 64-frame clips and extracts the per-temporal-token
representations, giving one (patch_tokens, pixel_frame) pair per temporal slot.

Architecture
-------------
  Linear projection:  1408 → 512  (per patch)
  Reshape to grid:    [B, 512, 24, 24]     (24 = sqrt(576))
  ConvTranspose  ×4:  24 → 384             (4-step upsampling)
  Output:             [B, 3, 384, 384]  in [0, 1]

~12M parameters.  Trains in ~1–2 hours on one A100 with 6×15 min of video.
"""

import glob
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoVideoProcessor


# ─── Config ────────────────────────────────────────────────────────────────────

HF_MODEL_NAME   = "facebook/vjepa2-vitg-fpc64-384"
# VIDEO_PATHS     = sorted(glob.glob('/project/3018078.02/MEG_ingmar/shorts/*.mp4'))
VIDEO_PATHS     = sorted(glob.glob('/project/3018078.02/MEG_ingmar/*.mp4'))

# Each clip encodes to 32 temporal tokens.  We sample CLIPS_PER_VIDEO
# non-overlapping 64-frame windows per video per epoch.

# TEST PARAMETERS
# CLIPS_PER_VIDEO = 1      # clip windows sampled per video; increase for longer videos
# CLIP_FRAMES     = 64      # must match model's fpc (frames-per-clip)
# TUBELET_SIZE    = 2       # raw frames per temporal token

# ACTUAL PARAMETERS
CLIPS_PER_VIDEO = 20      # clip windows sampled per video; increase for longer videos
CLIP_FRAMES     = 64      # must match model's fpc (frames-per-clip)
TUBELET_SIZE    = 2       # raw frames per temporal token

BATCH_SIZE      = 8       # (token, frame) pairs per gradient step
NUM_EPOCHS      = 40
LR              = 2e-4
OUTPUT_DIR      = Path('decoder_checkpoints')
SAVE_EVERY      = 5

# V-JEPA 2 model config for facebook/vjepa2-vitg-fpc64-384
N_SPATIAL       = 576     # spatial patches per temporal token (24×24)
EMBED_DIM       = 1408    # ViT-G embedding dimension
PATCH_GRID      = 24      # sqrt(N_SPATIAL)
DECODER_DIM     = 512     # decoder internal width (increase → sharper, slower)


# ─── Decoder architecture ──────────────────────────────────────────────────────

class VJepa2Decoder(nn.Module):
    """
    Convolutional upsampler: V-JEPA 2 patch tokens → pixel frame.

    Input : [B, N_SPATIAL, EMBED_DIM]   e.g. [B, 576, 1408]
    Output: [B, 3, img_size, img_size]  e.g. [B, 3, 384, 384]
    """
    def __init__(self, n_spatial=N_SPATIAL, embed_dim=EMBED_DIM,
                 decoder_dim=DECODER_DIM, img_size=384):
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


# ─── Clip loader ───────────────────────────────────────────────────────────────

def sample_clips_from_video(video_path, img_size, n_clips, clip_frames):
    """
    Sample up to `n_clips` non-overlapping windows of `clip_frames` raw frames
    from a video file.  Returns a list of clips, each a list of clip_frames
    H×W×3 uint8 RGB arrays.
    """
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total < clip_frames:
        return []

    # Sample start indices spread across the video
    max_start = total - clip_frames
    if n_clips == 1:
        starts = [0]
    else:
        starts = [int(i * max_start / (n_clips - 1)) for i in range(n_clips)]
    starts = sorted(set(starts))[:n_clips]

    clips = []
    for start in starts:
        cap   = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(clip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame  = cv2.resize(frame, (img_size, img_size))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) == clip_frames:
            clips.append(frames)

    return clips


# ─── Full-clip encoding ─────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_clip_to_token_frame_pairs(clip_frames_list, model, processor,
                                      clip_frames, tubelet_size, n_spatial):
    """
    Encode a single clip (list of clip_frames RGB uint8 arrays) with the frozen
    V-JEPA 2 encoder and return aligned (patch_token, pixel_frame) pairs.

    This is the CORRECT encoding approach: all frames are encoded together in
    one forward pass so temporal attention contextualises every token exactly as
    it will be at prediction time.

    Returns
    -------
    token_frames : list of (patch_tokens [N_spatial, embed_dim],
                             pixel_frame  [3, img_size, img_size]  float32 [0,1])
                   one entry per temporal token position
    """
    # Build tensor [clip_frames, C, H, W]  uint8 then encode as one clip
    video  = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1) for f in clip_frames_list
    ])                                                         # [T, C, H, W]
    inputs = processor(video, return_tensors="pt")
    pv     = inputs["pixel_values_videos"].cuda()             # [1, T, C, H, W]

    tokens_flat = model.get_vision_features(pv).cpu()         # [1, T_tok*P, D]
    T_tok = clip_frames // tubelet_size
    tokens = tokens_flat.squeeze(0).reshape(T_tok, n_spatial, -1)  # [T_tok, P, D]

    pairs = []
    for t in range(T_tok):
        # Match temporal token t to the first raw frame it covers
        frame_idx = t * tubelet_size
        frame_np  = clip_frames_list[frame_idx].astype(np.float32) / 255.0
        frame_t   = torch.from_numpy(frame_np).permute(2, 0, 1)   # [3, H, W]
        pairs.append((tokens[t], frame_t))                          # ([P, D], [3, H, W])

    return pairs


# ─── Dataset ───────────────────────────────────────────────────────────────────

class ClipTokenDataset(Dataset):
    """
    Pre-encodes all clip windows at construction time and stores the resulting
    (patch_tokens, pixel_frame) pairs in RAM.

    For very large video collections, replace with an on-disk cache or
    encode on-the-fly in a worker — but for 6×15 min this fits in CPU RAM.

    VRAM note: encoding is done on GPU once at init, then everything moves to CPU.
    """
    def __init__(self, video_paths, model, processor, img_size,
                 clips_per_video, clip_frames, tubelet_size, n_spatial):
        self.pairs = []
        total_clips = 0

        for vp in video_paths:
            clips = sample_clips_from_video(vp, img_size, clips_per_video, clip_frames)
            for clip in clips:
                pairs = encode_clip_to_token_frame_pairs(
                    clip, model, processor, clip_frames, tubelet_size, n_spatial)
                self.pairs.extend(pairs)
                total_clips += 1

        print(f"  Dataset: {len(self.pairs)} (token, frame) pairs "
              f"from {total_clips} clips across {len(video_paths)} videos")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        tokens, frame = self.pairs[i]
        return tokens, frame


# ─── Training ──────────────────────────────────────────────────────────────────

def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading V-JEPA 2 encoder: {HF_MODEL_NAME}")
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    encoder   = AutoModel.from_pretrained(HF_MODEL_NAME)
    encoder.cuda().eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    img_size  = processor.crop_size['height']
    print(f"  Encoder frozen. Crop size: {img_size}×{img_size}")

    print("Pre-encoding clips …")
    dataset    = ClipTokenDataset(
        VIDEO_PATHS, encoder, processor, img_size,
        CLIPS_PER_VIDEO, CLIP_FRAMES, TUBELET_SIZE, N_SPATIAL)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)

    decoder   = VJepa2Decoder(N_SPATIAL, EMBED_DIM, DECODER_DIM, img_size).cuda()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS)

    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"  Decoder parameters: {n_params/1e6:.1f}M")

    # ─── Checkpoint resuming ──────────────────────────────────────────────────
    start_epoch = 1
    best_loss = float('inf')
    resume_ckpt = OUTPUT_DIR / 'vjepa2_decoder_last.pt'
    
    if resume_ckpt.exists():
        print(f"  Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt)
        decoder.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"  → Resuming from epoch {start_epoch}, best_loss={best_loss:.4f}")
    else:
        print(f"  No checkpoint found, starting fresh training")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        decoder.train()
        epoch_loss = 0.0

        for batch_idx, (tokens, frames) in enumerate(dataloader):
            tokens = tokens.cuda()    # [B, N_spatial, D]
            frames = frames.cuda()    # [B, 3, H, W]

            recon = decoder(tokens)   # [B, 3, H, W]

            # L1 pixel loss
            loss_l1 = F.l1_loss(recon, frames)

            # Gradient-domain sharpness loss (proxy for perceptual quality)
            def grad_loss(a, b):
                gx = lambda t: t[..., 1:] - t[..., :-1]
                gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
                g  = frames.mean(dim=1, keepdim=True)
                r  = recon.mean(dim=1, keepdim=True)
                return F.mse_loss(gx(r), gx(g)) + F.mse_loss(gy(r), gy(g))
            loss_grad = grad_loss(recon, frames)

            loss = loss_l1 + 0.1 * loss_grad

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch}/{NUM_EPOCHS}  "
                      f"batch {batch_idx}/{len(dataloader)}  "
                      f"l1={loss_l1.item():.4f}  grad={loss_grad.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  avg_loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'state_dict':   decoder.state_dict(),
                'n_spatial':    N_SPATIAL,
                'embed_dim':    EMBED_DIM,
                'decoder_dim':  DECODER_DIM,
                'img_size':     img_size,
            }, OUTPUT_DIR / 'vjepa2_decoder_best.pt')
            print(f"  → New best saved (loss={best_loss:.4f})")

        # Save periodic checkpoint (for archival)
        if epoch % SAVE_EVERY == 0:
            torch.save({
                'state_dict':   decoder.state_dict(),
                'n_spatial':    N_SPATIAL,
                'embed_dim':    EMBED_DIM,
                'decoder_dim':  DECODER_DIM,
                'img_size':     img_size,
            }, OUTPUT_DIR / f'vjepa2_decoder_epoch{epoch:03d}.pt')
        
        # Always save resumable checkpoint (overwrites previous)
        torch.save({
            'epoch':        epoch,
            'state_dict':   decoder.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'best_loss':    best_loss,
            'n_spatial':    N_SPATIAL,
            'embed_dim':    EMBED_DIM,
            'decoder_dim':  DECODER_DIM,
            'img_size':     img_size,
        }, OUTPUT_DIR / 'vjepa2_decoder_last.pt')

    torch.save({
        'state_dict':  decoder.state_dict(),
        'n_spatial':   N_SPATIAL, 'embed_dim': EMBED_DIM,
        'decoder_dim': DECODER_DIM, 'img_size': img_size,
    }, OUTPUT_DIR / 'vjepa2_decoder_final.pt')
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"To use: load '{OUTPUT_DIR}/vjepa2_decoder_best.pt' in the prediction script.")


if __name__ == "__main__":
    train()
