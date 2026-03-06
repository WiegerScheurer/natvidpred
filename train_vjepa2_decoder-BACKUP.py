"""
Train a pixel decoder for V-JEPA 2 PREDICTOR output tokens
===========================================================
Maps V-JEPA 2 *predictor* output tokens  [B, 576, 1408]  →  pixel frames  [B, 3, 384, 384].

Why this differs from the encoder-token decoder
------------------------------------------------
The encoder produces spatially-diverse tokens: each of the 576 patch vectors
encodes its own local region of the frame.  The predictor, however, runs full
self-attention over all context + target positions and returns a *globally-mixed*
representation at each target position.  A decoder trained on encoder tokens
cannot handle this different distribution — it produces the 24×24 mosaic artifact.

This script trains the decoder on tokens that come from the predictor, so the
decoder learns the correct mapping for inference.

Data strategy for long, scarce videos
--------------------------------------
With 16 × ~12 min clips we have far more temporal diversity than frames.
We exploit this by generating many (context_window, target_position) pairs
per 64-frame clip window, with RANDOMISED context lengths:

  For each sampled 64-frame window:
    For each sampled target temporal position t  (MIN_CONTEXT_TOKENS ≤ t < T_tok):
      Sample context length  ctx_len  ~ Uniform[MIN_CONTEXT_TOKENS, t]
      Call predictor:  context tokens [t-ctx_len .. t-1]  →  predicted token at t
      Store:  (predicted_token [576, D],  pixel_frame_t  [3, H, W])

This gives TARGETS_PER_CLIP independent training pairs per clip window, each with
a different context window and target position, covering the full video densely.

Varying ctx_len is important: it forces the decoder to handle predictor outputs
that range from short-context (high uncertainty, globally smooth) to long-context
(low uncertainty, more spatially structured), making it robust at inference time.

Architecture (identical to the encoder-decoder, just trained differently)
--------------------------------------------------------------------------
  Linear projection:  1408 → 512  (per patch)
  Reshape to grid:    [B, 512, 24, 24]
  ConvTranspose  ×4:  24 → 384
  Output:             [B, 3, 384, 384]  in [0, 1]
"""

import glob
import random
import sys
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
# VIDEO_PATHS     = [vid if not "bw" in vid for vid in sorted(glob.glob('/project/3018078.02/MEG_ingmar/*.mp4'))]
VIDEO_PATHS     = [vid for vid in sorted(glob.glob('/project/3018078.02/MEG_ingmar/*.mp4')) if "bw" not in vid] # only take forward vids

CLIP_FRAMES     = 64      # must match model's fpc
TUBELET_SIZE    = 2       # raw frames per temporal token  →  T_tok = 64/2 = 32
N_SPATIAL       = 576     # spatial patches per temporal token (24×24)
EMBED_DIM       = 1408    # ViT-G embedding dimension
PATCH_GRID      = 24      # sqrt(N_SPATIAL)
DECODER_DIM     = 512     # decoder internal width

# ── Data sampling ─────────────────────────────────────────────────────────────
# Long videos → sample many clip windows per video.
# ~12 min @ 25 fps = ~18 000 frames → ~281 non-overlapping 64-frame windows.
# Using 80 gives good coverage without re-encoding the same frames too often.

EASY_MODE_PREDICTOR = True  # Set to True for decoder-only evaluation
# Easy mode uses EVERY frame: context=[frame_t] → predict frame_t (identity mapping).
# This maximizes data (32 pairs/clip instead of 12) and isolates decoder quality—
# no predictor uncertainty, just clean token→pixel learning.
# At inference, the predictor will predict actual future frames, but the decoder
# will have learned the mapping on maximally clean examples.

# PRODUCTION PARAMETERS (slow, thorough):
# CLIPS_PER_VIDEO     = 50 #
# TARGETS_PER_CLIP    = 12

# FAST TEST PARAMETERS (use these first to verify the pipeline works):
CLIPS_PER_VIDEO     = 2    # 16 videos × 2 clips = 32 encoding passes (~3–5 min)
TARGETS_PER_CLIP    = 3    # 32 clips × 3 targets = 96 pairs (small but workable)

# Per clip window, sample this many (context, target) predictor pairs.
# Each pair uses a freshly randomised context length and target position.

# Context length range.  Keeping MIN low (2–3) forces the decoder to handle
# high-uncertainty predictor outputs; MAX caps at half the token sequence.
MIN_CONTEXT_TOKENS  = 3
MAX_CONTEXT_TOKENS  = 16   # ≤ T_tok - 1  (= 31 for fpc64)

# ── Training ──────────────────────────────────────────────────────────────────
# FAST TEST (verify pipeline works in ~5–10 min total):
BATCH_SIZE      = 2
NUM_EPOCHS      = 1
SAVE_EVERY      = 1

# PRODUCTION (uncomment after successful test, expects ~2–3 hours):
# BATCH_SIZE      = 8
# NUM_EPOCHS      = 40
LR              = 2e-4
OUTPUT_DIR      = Path('decoder_checkpoints_predictor_test')
# SAVE_EVERY      = 5

# LR              = 2e-4
# OUTPUT_DIR      = Path('decoder_checkpoints_predictor')


# ─── Decoder architecture ─────────────────────────────────────────────────────

class VJepa2Decoder(nn.Module):
    """
    Convolutional upsampler: V-JEPA 2 patch tokens → pixel frame.

    Input : [B, N_SPATIAL, EMBED_DIM]
    Output: [B, 3, img_size, img_size]
    """
    def __init__(self, n_spatial=N_SPATIAL, embed_dim=EMBED_DIM,
                 decoder_dim=DECODER_DIM, img_size=384):
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
        x = self.proj(patch_tokens)                              # [B, P, D_dec]
        x = x.permute(0, 2, 1)                                   # [B, D_dec, P]
        x = x.reshape(B, -1, self.patch_grid, self.patch_grid)   # [B, D_dec, 24, 24]
        x = self.up(x)                                            # [B, 3, H, W]
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, (self.img_size, self.img_size),
                               mode='bilinear', align_corners=False)
        return x


# ─── Clip loader ──────────────────────────────────────────────────────────────

def sample_clip_starts(video_path, n_clips, clip_frames):
    """Return up to n_clips evenly-spaced start frame indices."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < clip_frames:
        return []
    max_start = total - clip_frames
    if n_clips == 1:
        return [0]
    starts = [int(i * max_start / (n_clips - 1)) for i in range(n_clips)]
    return sorted(set(starts))[:n_clips]


def load_clip(video_path, start_frame, clip_frames, img_size):
    """Load exactly clip_frames consecutive RGB frames starting at start_frame."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(clip_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(
            cv2.resize(frame, (img_size, img_size)), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames if len(frames) == clip_frames else None


# ─── Predictor-based pair generation ─────────────────────────────────────────

@torch.inference_mode()
def call_predictor(model, encoder_hidden_states, ctx_start, ctx_len,
                   target_t, n_spatial):
    """
    Ask the V-JEPA 2 predictor to predict the token at temporal position
    `target_t`, using context tokens at positions [ctx_start .. ctx_start+ctx_len-1].

    Parameters
    ----------
    encoder_hidden_states : [1, T_tok*P, D]  full encoded sequence (on GPU)
    ctx_start  : first context temporal position (int)
    ctx_len    : number of context temporal positions (int)
    target_t   : target temporal position to predict (int)
    n_spatial  : P = 576

    Returns
    -------
    predicted_token : [n_spatial, D]  CPU tensor, or None if predictor unavailable
    """
    device  = encoder_hidden_states.device
    seq_len = encoder_hidden_states.shape[1]
    D       = encoder_hidden_states.shape[-1]

    predictor = getattr(model, 'predictor', None)
    if predictor is None:
        return None

    ctx_mask = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    ctx_mask[:, ctx_start * n_spatial:(ctx_start + ctx_len) * n_spatial] = 1

    tgt_mask = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    tgt_mask[:, target_t * n_spatial:(target_t + 1) * n_spatial] = 1

    try:
        out = predictor(
            encoder_hidden_states=encoder_hidden_states,
            context_mask=[ctx_mask],
            target_mask=[tgt_mask],
        )
        # out.last_hidden_state : [1, seq_len, D]
        tok = out.last_hidden_state[0,
              target_t * n_spatial:(target_t + 1) * n_spatial, :]  # [P, D]
        return tok.cpu()
    except Exception as e:
        print(f"  Predictor call failed: {e}")
        return None


@torch.inference_mode()
def generate_predictor_pairs(clip_frames_list, model, processor,
                              clip_frames, tubelet_size, n_spatial,
                              targets_per_clip, min_context, max_context,
                              easy_mode=False): 
    """
    Encode one 64-frame clip and generate predictor-token training pairs.

    Easy mode (decoder evaluation)
    -------------------------------
    Generate T_tok pairs, one for each temporal position:
      - Context = [frame t], Target = [frame t]
      - Predictor sees the frame and "predicts" the same frame (trivial task)
      - Output tokens are maximally clean predictor-style representations
      - Decoder learns token→pixel mapping without predictor uncertainty
      - Uses ALL frames: 32 pairs per clip instead of 12

    Hard mode (general training)
    -----------------------------
    Sample targets_per_clip pairs with randomised context windows:
      - Pick random target_t in [min_context, T_tok)
      - Pick random context length and position
      - Predictor makes non-trivial predictions
      - Decoder learns to handle uncertain predictor outputs

    Returns list of (predicted_token [P, D], pixel_frame [3, H, W]) pairs.
    """
    T_tok = clip_frames // tubelet_size  # e.g. 32

    # ── Encode full clip once ─────────────────────────────────────────────────
    video  = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1) for f in clip_frames_list
    ])                                                         # [T, C, H, W]
    inputs = processor(video, return_tensors="pt")
    pv     = inputs["pixel_values_videos"].cuda()
    tokens_flat = model.get_vision_features(pv)               # [1, T_tok*P, D]  on GPU

    pairs = []

    if easy_mode:
        # EASY MODE: use EVERY frame, context = target (identity mapping)
        for target_t in range(T_tok):
            ctx_start = target_t
            ctx_len = 1  # just this frame itself
            
            pred_tok = call_predictor(
                model, tokens_flat, ctx_start, ctx_len, target_t, n_spatial)
            if pred_tok is None:
                continue
            
            # GT pixel frame for target_t
            frame_idx = min(target_t * tubelet_size, len(clip_frames_list) - 1)
            frame_np  = clip_frames_list[frame_idx].astype(np.float32) / 255.0
            frame_t   = torch.from_numpy(frame_np).permute(2, 0, 1)  # [3, H, W]
            
            pairs.append((pred_tok, frame_t))
    else:
        # HARD MODE: randomized context windows
        max_attempts = targets_per_clip * 4
        attempts = 0
        
        while len(pairs) < targets_per_clip and attempts < max_attempts:
            attempts += 1
            
            if T_tok <= min_context:
                break
            target_t = random.randint(min_context, T_tok - 1)
            
            max_ctx  = min(max_context, target_t)
            ctx_len  = random.randint(min_context, max_ctx)
            max_gap  = target_t - ctx_len
            gap      = random.randint(0, max_gap)
            ctx_start = target_t - ctx_len - gap
            ctx_start = max(0, ctx_start)
            
            pred_tok = call_predictor(
                model, tokens_flat, ctx_start, ctx_len, target_t, n_spatial)
            if pred_tok is None:
                continue
            
            # GT pixel frame for target_t
            frame_idx = min(target_t * tubelet_size, len(clip_frames_list) - 1)
            frame_np  = clip_frames_list[frame_idx].astype(np.float32) / 255.0
            frame_t   = torch.from_numpy(frame_np).permute(2, 0, 1)  # [3, H, W]
            
            pairs.append((pred_tok, frame_t))

    return pairs

# ─── Dataset ──────────────────────────────────────────────────────────────────

class PredictorTokenDataset(Dataset):
    """
    Pre-generates all (predictor_token, pixel_frame) pairs at construction time.

    For very large collections, replace with an on-disk cache.  With 16 videos
    and 80 clips each this stays manageable in CPU RAM (~20–30 GB for the tokens;
    use CLIPS_PER_VIDEO / TARGETS_PER_CLIP to tune memory vs. coverage).
    """
    def __init__(self, video_paths, model, processor, img_size,
                 clips_per_video, clip_frames, tubelet_size, n_spatial,
                 targets_per_clip, min_context, max_context, easy_mode=False):
        self.pairs  = []
        n_clips_ok  = 0
        n_pred_fail = 0

        for vi, vp in enumerate(video_paths):
            starts = sample_clip_starts(vp, clips_per_video, clip_frames)
            print(f"  Video {vi+1}/{len(video_paths)}: "
                  f"{Path(vp).name}  —  {len(starts)} clip windows")
            sys.stdout.flush()

            for ci, start in enumerate(starts):
                frames = load_clip(vp, start, clip_frames, img_size)
                if frames is None:
                    continue

                new_pairs = generate_predictor_pairs(
                    frames, model, processor,
                    clip_frames, tubelet_size, n_spatial,
                    targets_per_clip, min_context, max_context, 
                    easy_mode=easy_mode) # ? add this?

                if len(new_pairs) == 0:
                    n_pred_fail += 1
                else:
                    self.pairs.extend(new_pairs)
                    n_clips_ok += 1
                
                # Progress every 10 clips
                if (ci + 1) % 10 == 0:
                    print(f"    → Clip {ci+1}/{len(starts)} processed, "
                          f"{len(self.pairs)} pairs so far")
                    sys.stdout.flush()

        print(f"\n  Dataset ready: {len(self.pairs)} (predictor_token, frame) pairs  "
              f"from {n_clips_ok} clips  ({n_pred_fail} clips had predictor failures)")
        sys.stdout.flush()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tokens, frame = self.pairs[idx]
        return tokens, frame


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading V-JEPA 2: {HF_MODEL_NAME}")
    sys.stdout.flush()
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    encoder   = AutoModel.from_pretrained(HF_MODEL_NAME)
    encoder.cuda().eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    img_size = processor.crop_size['height']
    print(f"  Encoder + predictor frozen. Crop size: {img_size}×{img_size}")
    print(f"  Context range: [{MIN_CONTEXT_TOKENS}, {MAX_CONTEXT_TOKENS}] tokens")
    print(f"  Clips per video: {CLIPS_PER_VIDEO}  |  "
          f"Easy mode: {EASY_MODE_PREDICTOR}")
    
    if EASY_MODE_PREDICTOR:
        T_tok = CLIP_FRAMES // TUBELET_SIZE
        expected_pairs = len(VIDEO_PATHS) * CLIPS_PER_VIDEO * T_tok
        print(f"  Expected pairs (easy mode, all frames): ~{expected_pairs:,}")
    else:
        expected_pairs = len(VIDEO_PATHS) * CLIPS_PER_VIDEO * TARGETS_PER_CLIP
        print(f"  Expected pairs (hard mode): ~{expected_pairs:,}")
    sys.stdout.flush()

    print("\nGenerating predictor training pairs …")
    sys.stdout.flush()
    
    dataset = PredictorTokenDataset(
        VIDEO_PATHS, encoder, processor, img_size,
        CLIPS_PER_VIDEO, CLIP_FRAMES, TUBELET_SIZE, N_SPATIAL,
        TARGETS_PER_CLIP, MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS,
        easy_mode=EASY_MODE_PREDICTOR)

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty.  The predictor API call failed for all clips.  "
            "Check that model.predictor exists and that call_predictor() is "
            "using the correct signature for your V-JEPA 2 version.")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)

    decoder   = VJepa2Decoder(N_SPATIAL, EMBED_DIM, DECODER_DIM, img_size).cuda()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS)

    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"  Decoder parameters: {n_params/1e6:.1f}M")

    # ── Checkpoint resuming ───────────────────────────────────────────────────
    start_epoch = 1
    best_loss   = float('inf')
    resume_ckpt = OUTPUT_DIR / 'vjepa2_decoder_predictor_last.pt'

    if resume_ckpt.exists():
        print(f"  Resuming from: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location='cpu')
        decoder.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        print(f"  → Epoch {start_epoch}, best_loss={best_loss:.4f}")
    else:
        print("  No checkpoint found, starting fresh.")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        decoder.train()
        epoch_loss = 0.0

        for batch_idx, (tokens, frames) in enumerate(dataloader):
            tokens = tokens.cuda()   # [B, P, D]
            frames = frames.cuda()   # [B, 3, H, W]

            recon = decoder(tokens)  # [B, 3, H, W]

            # L1 pixel loss
            loss_l1 = F.l1_loss(recon, frames)

            # Gradient-domain sharpness loss
            def grad_loss():
                gx = lambda t: t[..., 1:]  - t[..., :-1]
                gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
                g  = frames.mean(dim=1, keepdim=True)
                r  = recon.mean(dim=1, keepdim=True)
                return F.mse_loss(gx(r), gx(g)) + F.mse_loss(gy(r), gy(g))


            # OLD (too simple):
            # loss = loss_l1 + 0.1 * grad_loss()

            # Better (with perceptual loss added, mits lpips available):
            try:
                import lpips
                lpips_fn = lpips.LPIPS(net='vgg').cuda()
                loss_perceptual = lpips_fn(recon, frames).mean()
                loss = loss_l1 + 0.1 * loss_perceptual
            except ImportError:
                loss = loss_l1  # fallback

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch}/{NUM_EPOCHS}  "
                      f"batch {batch_idx}/{len(dataloader)}  "
                      f"l1={loss_l1.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  avg_loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'state_dict':        decoder.state_dict(),
                'n_spatial':         N_SPATIAL,
                'embed_dim':         EMBED_DIM,
                'decoder_dim':       DECODER_DIM,
                'img_size':          img_size,
                'trained_on':        'predictor_tokens',
                'min_context':       MIN_CONTEXT_TOKENS,
                'max_context':       MAX_CONTEXT_TOKENS,
            }, OUTPUT_DIR / 'vjepa2_decoder_predictor_best.pt')
            print(f"  → New best saved (loss={best_loss:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                'state_dict':  decoder.state_dict(),
                'n_spatial':   N_SPATIAL, 'embed_dim': EMBED_DIM,
                'decoder_dim': DECODER_DIM, 'img_size': img_size,
            }, OUTPUT_DIR / f'vjepa2_decoder_predictor_epoch{epoch:03d}.pt')

        torch.save({
            'epoch':         epoch,
            'state_dict':    decoder.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_loss':     best_loss,
            'n_spatial':     N_SPATIAL,
            'embed_dim':     EMBED_DIM,
            'decoder_dim':   DECODER_DIM,
            'img_size':      img_size,
            'trained_on':    'predictor_tokens',
        }, OUTPUT_DIR / 'vjepa2_decoder_predictor_last.pt')

    torch.save({
        'state_dict':  decoder.state_dict(),
        'n_spatial':   N_SPATIAL, 'embed_dim': EMBED_DIM,
        'decoder_dim': DECODER_DIM, 'img_size': img_size,
        'trained_on':  'predictor_tokens',
    }, OUTPUT_DIR / 'vjepa2_decoder_predictor_final.pt')

    print(f"\nTraining complete.  Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {OUTPUT_DIR}/vjepa2_decoder_predictor_best.pt")
    print("Update DECODER_CHECKPOINT in the prediction script to point here.")


if __name__ == "__main__":
    train()







