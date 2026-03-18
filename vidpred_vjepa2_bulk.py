import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor

# --- Configuration ---
HF_MODEL_NAME   = "facebook/vjepa2-vitg-fpc64-384"
CONTEXT_STEPS   = 10   # Number of latent steps for context
HORIZONS        = [4, 8, 12] # Predict n steps into the future
TUBELET_SIZE    = 2    
STRIDE          = 2    
OUTPUT_DIR      = Path("vjepa_sliding_results")
VIDEO_DIR       = Path("/project/3018078.02/MEG_ingmar/shorts/")

def get_low_level_stats(frame, prev_frame=None):
    """Computes RMS contrast and Mean Optical Flow magnitude."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    rms_contrast = np.std(gray)
    
    flow_mag = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mag = np.mean(mag)
        
    return rms_contrast, flow_mag

@torch.inference_mode()
# def extract_all_latents(model, processor, frames, batch_size=40):
#     """Extracts latents for the video in batches to avoid CUDA OOM and patch size mismatch."""
#     latents_list = []
#     device = next(model.parameters()).device
#     num_frames = len(frames)
#     # Only process up to the largest multiple of TUBELET_SIZE
#     usable_frames = (num_frames // TUBELET_SIZE) * TUBELET_SIZE
#     frames = frames[:usable_frames]
#     num_steps = len(frames) // TUBELET_SIZE
#     for i in range(0, len(frames), batch_size):
#         batch_frames = frames[i:i+batch_size]
#         # Drop incomplete tubelet at the end of the batch
#         if len(batch_frames) < TUBELET_SIZE:
#             break
#         # Also ensure batch size is a multiple of TUBELET_SIZE
#         batch_usable = (len(batch_frames) // TUBELET_SIZE) * TUBELET_SIZE
#         batch_frames = batch_frames[:batch_usable]
#         if not batch_frames:
#             continue
#         video = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in batch_frames])
#         inputs = processor(video, return_tensors="pt")
#         pixel_values = inputs["pixel_values_videos"].to(device)
#         feats = model.get_vision_features(pixel_values).cpu()
#         latents_list.append(feats)
#         torch.cuda.empty_cache()
#     # Now, check all feats have the same shape except for dim 0
#     shapes = [x.shape[1] for x in latents_list]
#     if len(set(shapes)) > 1:
#         min_patches = min(shapes)
#         latents_list = [x[:, :min_patches, :] for x in latents_list]
#     feats = torch.cat(latents_list, dim=0)
#     P = feats.shape[1] // num_steps
#     return feats.reshape(num_steps, P, -1)

@torch.inference_mode()
def extract_all_latents(model, processor, frames):
    """Extracts latents for the video tubelet-by-tubelet to avoid shape mismatches."""
    device = next(model.parameters()).device
    num_frames = len(frames)
    num_steps = num_frames // TUBELET_SIZE
    latents_list = []
    min_patches = None

    for i in range(num_steps):
        tubelet_frames = frames[i * TUBELET_SIZE : (i + 1) * TUBELET_SIZE]
        video = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in tubelet_frames])
        inputs = processor(video.unsqueeze(0), return_tensors="pt")  # Add batch dim for video
        pixel_values = inputs["pixel_values_videos"].to(device)
        feats = model.get_vision_features(pixel_values).cpu().squeeze(0)  # Remove batch dim
        latents_list.append(feats)
        if min_patches is None or feats.shape[0] < min_patches:
            min_patches = feats.shape[0]

    # Truncate all to min_patches to ensure stackability
    latents_list = [f[:min_patches, :] for f in latents_list]
    feats = torch.stack(latents_list)  # [num_steps, P, D]
    return feats

# def extract_all_latents(model, processor, frames, batch_size=40):
#     """Extracts latents for the video in batches to avoid CUDA OOM and patch size mismatch."""
#     latents_list = []
#     device = next(model.parameters()).device
#     num_frames = len(frames)
#     usable_frames = (num_frames // TUBELET_SIZE) * TUBELET_SIZE
#     frames = frames[:usable_frames]
#     num_steps = len(frames) // TUBELET_SIZE
#     expected_patches = None
#     for i in range(0, len(frames), batch_size):
#         batch_frames = frames[i:i+batch_size]
#         # Drop incomplete tubelet at the end of the batch
#         if len(batch_frames) < TUBELET_SIZE:
#             break
#         batch_usable = (len(batch_frames) // TUBELET_SIZE) * TUBELET_SIZE
#         batch_frames = batch_frames[:batch_usable]
#         if not batch_frames:
#             continue
#         video = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in batch_frames])
#         inputs = processor(video, return_tensors="pt")
#         pixel_values = inputs["pixel_values_videos"].to(device)
#         feats = model.get_vision_features(pixel_values).cpu()
#         # Set expected_patches from first batch
#         if expected_patches is None:
#             expected_patches = feats.shape[1] // (len(batch_frames) // TUBELET_SIZE)
#         # Truncate to expected_patches if needed
#         steps_in_batch = len(batch_frames) // TUBELET_SIZE
#         if feats.shape[1] != steps_in_batch * expected_patches:
#             min_patches = min(feats.shape[1] // steps_in_batch, expected_patches)
#             feats = feats[:, :min_patches * steps_in_batch, :]
#         latents_list.append(feats)
#         torch.cuda.empty_cache()
#     feats = torch.cat(latents_list, dim=0)
#     # Now, reshape using the robust patch count
#     P = expected_patches
#     return feats.reshape(num_steps, P, -1)


@torch.inference_mode()
def predict_at_horizon(model, context_latents, horizon):
    """Predicts a specific number of steps into the future."""
    T_ctx, P, D = context_latents.shape
    total_len = T_ctx + horizon
    device = next(model.parameters()).device
    
    ctx_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
    ctx_mask[:, :T_ctx * P] = 1
    tgt_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
    tgt_mask[:, T_ctx * P:] = 1

    z_padded = torch.zeros(1, total_len * P, D, device=device)
    z_padded[:, :T_ctx * P, :] = context_latents.reshape(1, -1, D).to(device)

    out = model.predictor(encoder_hidden_states=z_padded, context_mask=[ctx_mask], target_mask=[tgt_mask])
    return out.last_hidden_state[:, T_ctx * P :, :].reshape(horizon, P, D).cpu()

def process_video(video_path, model, processor):
    print(f"Processing: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    raw_frames = []
    low_level_stats = []
    prev_f = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb)
        
        # Low level stats
        rms, flow = get_low_level_stats(frame_rgb, prev_f)
        low_level_stats.append({"rms_contrast": rms, "optical_flow": flow})
        prev_f = frame_rgb
    cap.release()

    # Get V-JEPA Latents for the whole video
    sampled_frames = raw_frames[::STRIDE]
    all_latents = extract_all_latents(model, processor, sampled_frames)
    
    results = []
    num_steps = all_latents.shape[0]
    
    # Sliding Window
    for i in range(num_steps - max(HORIZONS) - CONTEXT_STEPS):
        context = all_latents[i : i + CONTEXT_STEPS]
        row = {"latent_step": i, "raw_frame_idx": i * TUBELET_SIZE * STRIDE}
        
        # Add low-level stats (aligned to the end of the context window)
        idx = min(row["raw_frame_idx"] + (CONTEXT_STEPS * TUBELET_SIZE * STRIDE), len(low_level_stats)-1)
        row.update(low_level_stats[idx])

        for h in HORIZONS:
            pred = predict_at_horizon(model, context, h)
            target = all_latents[i + CONTEXT_STEPS : i + CONTEXT_STEPS + h]
            
            # Cosine Distance
            cos_sim = F.cosine_similarity(pred.mean(1), target.mean(1), dim=-1)
            row[f"unpredictability_h{h}_cosine"] = (1.0 - cos_sim).mean().item()
            
            # Euclidean Distance
            eucl_dist = torch.norm(pred.mean(1) - target.mean(1), dim=-1)
            row[f"unpredictability_h{h}_euclidean"] = eucl_dist.mean().item()
            
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / f"{video_path.stem}_metrics.csv", index=False)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()

    for v_file in VIDEO_DIR.glob("longclip_part42_bw.mp4"):
        try:
            process_video(v_file, mdl, proc)
        except Exception as e:
            print(f"Failed {v_file.name}: {e}")



# import os
# import cv2
# import torch
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import torch.nn.functional as F
# from transformers import AutoModel, AutoVideoProcessor

# # --- Configuration ---
# HF_MODEL_NAME   = "facebook/vjepa2-vitg-fpc64-384"
# CONTEXT_STEPS   = 10   # Number of latent steps for context
# HORIZONS        = [4, 8, 12] # Predict n steps into the future
# TUBELET_SIZE    = 2    
# STRIDE          = 2    
# OUTPUT_DIR      = Path("vjepa_sliding_results")
# VIDEO_DIR       = Path("/project/3018078.02/MEG_ingmar/shorts/")

# def get_low_level_stats(frame, prev_frame=None):
#     """Computes RMS contrast and Mean Optical Flow magnitude."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     rms_contrast = np.std(gray)
    
#     flow_mag = 0.0
#     if prev_frame is not None:
#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         flow_mag = np.mean(mag)
        
#     return rms_contrast, flow_mag

# @torch.inference_mode()
# def extract_all_latents(model, processor, frames):
#     """Extracts latents for the entire video at once to save time."""
#     video = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in frames])
#     inputs = processor(video, return_tensors="pt")
#     pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
#     feats = model.get_vision_features(pixel_values).cpu()
#     # Reshape to [Steps, Patches, Dim]
#     num_steps = len(frames) // TUBELET_SIZE
#     P = feats.shape[1] // num_steps
#     return feats.reshape(num_steps, P, -1)

# @torch.inference_mode()
# def predict_at_horizon(model, context_latents, horizon):
#     """Predicts a specific number of steps into the future."""
#     T_ctx, P, D = context_latents.shape
#     total_len = T_ctx + horizon
#     device = next(model.parameters()).device
    
#     ctx_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
#     ctx_mask[:, :T_ctx * P] = 1
#     tgt_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
#     tgt_mask[:, T_ctx * P:] = 1

#     z_padded = torch.zeros(1, total_len * P, D, device=device)
#     z_padded[:, :T_ctx * P, :] = context_latents.reshape(1, -1, D).to(device)

#     out = model.predictor(encoder_hidden_states=z_padded, context_mask=[ctx_mask], target_mask=[tgt_mask])
#     return out.last_hidden_state[:, T_ctx * P :, :].reshape(horizon, P, D).cpu()

# def process_video(video_path, model, processor):
#     print(f"Processing: {video_path.name}")
#     cap = cv2.VideoCapture(str(video_path))
#     raw_frames = []
#     low_level_stats = []
#     prev_f = None
    
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame_rgb = cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB)
#         raw_frames.append(frame_rgb)
        
#         # Low level stats
#         rms, flow = get_low_level_stats(frame_rgb, prev_f)
#         low_level_stats.append({"rms_contrast": rms, "optical_flow": flow})
#         prev_f = frame_rgb
#     cap.release()

#     # Get V-JEPA Latents for the whole video
#     # We use STRIDE to match your previous logic
#     sampled_frames = raw_frames[::STRIDE]
#     all_latents = extract_all_latents(model, processor, sampled_frames)
    
#     results = []
#     num_steps = all_latents.shape[0]
    
#     # Sliding Window
#     # Each 'step' in latent space covers TUBELET_SIZE * STRIDE raw frames
#     for i in range(num_steps - max(HORIZONS) - CONTEXT_STEPS):
#         context = all_latents[i : i + CONTEXT_STEPS]
#         row = {"latent_step": i, "raw_frame_idx": i * TUBELET_SIZE * STRIDE}
        
#         # Add low-level stats (aligned to the end of the context window)
#         idx = min(row["raw_frame_idx"] + (CONTEXT_STEPS * TUBELET_SIZE * STRIDE), len(low_level_stats)-1)
#         row.update(low_level_stats[idx])

#         for h in HORIZONS:
#             pred = predict_at_horizon(model, context, h)
#             target = all_latents[i + CONTEXT_STEPS : i + CONTEXT_STEPS + h]
            
#             # Metric: Mean Cosine Distance across the horizon
#             cos_sim = F.cosine_similarity(pred.mean(1), target.mean(1), dim=-1)
#             row[f"unpredictability_h{h}"] = (1.0 - cos_sim).mean().item()
            
#         results.append(row)

#     df = pd.DataFrame(results)
#     df.to_csv(OUTPUT_DIR / f"{video_path.stem}_metrics.csv", index=False)

# if __name__ == "__main__":
#     OUTPUT_DIR.mkdir(exist_ok=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
#     mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()

#     # for v_file in VIDEO_DIR.glob("*.mp4"):
#     for v_file in VIDEO_DIR.glob("longclip_part42_bw.mp4"):
#         try:
#             process_video(v_file, mdl, proc)
#         except Exception as e:
#             print(f"Failed {v_file.name}: {e}")