"""
V-JEPA2 Future Frame Prediction Analysis
=========================================

WHAT THIS SCRIPT DOES:
1. Loads videos and extracts frames from the middle section
2. Uses V-JEPA2 to compute latent representations (compact feature vectors)
3. Predicts future latent vectors from only context frames (causal prediction)
4. Compares predictions to ground truth and measures prediction error

DETAILED FLOW:
- Context Window: Takes N frames from the past (e.g., 10 frames)
  → Encodes them with V-JEPA2 to get "context latents"
  → Uses these as input to a predictor model
  
- Prediction: The predictor generates M frames into the future (e.g., 15 frames)
  → Only sees context frames, cannot cheat by looking at future
  → Outputs predicted latents for future time steps
  
- Ground Truth: Takes the actual M future frames
  → Processes them independently with V-JEPA2
  → Gets ground truth latents (what actually happened)
  
- Comparison: Measures how close predicted latents are to ground truth
  → Cosine distance (0=identical direction, 1=opposite)
  → L2/Euclidean distance (actual magnitude difference)
  
- Visualization: Creates GIFs showing
  → The video frame at each timestep
  → Prediction error curves as the model predicts further into future

WHY THIS MATTERS:
- Tests if V-JEPA2's predictor can extrapolate motion/changes
- Identifies when predictions break down (error grows with time)
- Bidirectional analysis: predict forward AND backward to check consistency
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor


# --- Configuration ---
HF_MODEL_NAME     = "facebook/vjepa2-vitg-fpc64-384"
CONTEXT_TOKENS    = 10   
PREDICT_TOKENS    = 15   
TUBELET_SIZE      = 2    
STRIDE            = 3  
OUTPUT_BASE_DIR   = Path("vjepa_results_v4")
# VIDEO_NAMES       = ["deurinhuis", "trap", "titanic_bound", "iglo", "kanon", "sprongwagon", "geiser", 
                    #  "klepdonder", "gevelstort", "foetsie", "uitzichtdoek", "yenga", "whackamole", 
                    #  "trein_portret", "gooi_krat", "bw_testclip_pluim", "bw_testclip_lazeren"]
VIDEO_NAMES      = ["geiser"]  # For quick testing
import matplotlib
matplotlib.use('Agg')


# --- Helper Functions ---


def calculate_metrics(predicted, ground_truth):
    """
    Compute distance metrics between predicted and ground truth latents.
    
    Args:
        predicted: [T, P, D] predicted latents
        ground_truth: [T, P, D] ground truth latents
    
    Returns:
        cos_dist: cosine distance per timestep (0=identical direction, 1=opposite)
        l2_dist: L2/Euclidean distance per timestep
    """
    # Average across spatial patches for each timestep
    p_vec = predicted.mean(dim=1)  # [T, D]
    g_vec = ground_truth.mean(dim=1)  # [T, D]
    
    # Handle potential length mismatch
    min_len = min(p_vec.size(0), g_vec.size(0))
    p_vec, g_vec = p_vec[:min_len], g_vec[:min_len]
    
    # Compute distances
    cos_sim = F.cosine_similarity(p_vec, g_vec, dim=-1)
    cos_dist = (1.0 - cos_sim).numpy()  # Convert similarity to distance
    l2_dist = torch.norm(p_vec - g_vec, p=2, dim=-1).numpy()
    
    return cos_dist, l2_dist


@torch.inference_mode()
def get_vjepa_latents(model, processor, frames):
    """
    Encode video frames into V-JEPA2 latent space.
    
    Args:
        model: V-JEPA2 model
        processor: V-JEPA2 video processor
        frames: List of numpy arrays [H, W, 3] in RGB
    
    Returns:
        feats: [T_out, P, D] where T_out = len(frames)//TUBELET_SIZE
               - T_out: temporal tokens (tubelet groups)
               - P: spatial patches
               - D: feature dimension
    """
    # Stack frames and convert to tensor format [T, C, H, W]
    video = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames])
    
    # Process through V-JEPA2
    inputs = processor(video, return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
    feats = model.get_vision_features(pixel_values).cpu()
    
    # Reshape to [T_tokens, P_spatial, D_feature]
    P = feats.shape[1] // (len(frames) // TUBELET_SIZE)
    return feats.reshape(-1, P, feats.shape[-1])


@torch.inference_mode()
def predict_future_latents(model, context_latents, num_future_steps):
    """
    Predict future latents using the model's predictor.
    CRITICAL: Only context latents are visible to the predictor (causal prediction).
    
    Args:
        model: V-JEPA2 model with predictor component
        context_latents: [T_ctx, P, D] latents from context frames only
        num_future_steps: Number of future timesteps to predict
    
    Returns:
        predicted_latents: [num_future_steps, P, D] predicted future latents
    """
    T_ctx, P, D = context_latents.shape
    total_len = T_ctx + num_future_steps
    predictor = model.predictor
    device = next(model.parameters()).device
    
    # Create masks:
    # - context_mask: marks which positions contain actual context data
    # - target_mask: marks which positions should be predicted
    # NOTE: A value of 1 typically means "this position is valid/visible to the model"
    #       The predictor should only see context (1) and predict target positions (0)
    
    context_mask = torch.ones(1, T_ctx * P, dtype=torch.int64, device=device)
    # Pad with zeros for future positions (model cannot see/use these)
    context_mask = torch.cat([
        context_mask,
        torch.zeros(1, num_future_steps * P, dtype=torch.int64, device=device)
    ], dim=1)
    
    target_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
    target_mask[:, T_ctx * P : (T_ctx + num_future_steps) * P] = 1
    
    # Build padded sequence: [context latents | zeros for future]
    z_padded = torch.zeros(1, total_len * P, D, device=device)
    z_ctx = context_latents.reshape(1, -1, D).to(device)
    z_padded[:, :T_ctx * P, :] = z_ctx
    
    # Run predictor - generates values for target positions
    out = predictor(
        encoder_hidden_states=z_padded,
        context_mask=[context_mask],  # Only context is visible
        target_mask=[target_mask]      # Predict these positions
    )
    
    # Extract only the predicted future portion and reshape
    return out.last_hidden_state[:, T_ctx * P :, :].reshape(num_future_steps, P, D).cpu()


def run_analysis(model, processor, video_path):
    """
    Run full analysis pipeline on a single video.
    
    For both forward and backward prediction:
    1. Extract context frames (past) and target frames (future)
    2. Encode context frames → context latents
    3. Encode target frames independently → ground truth latents
    4. Predict future latents from context only
    5. Measure prediction error
    6. Create visualization GIF
    """
    video_stem = Path(video_path).stem
    video_dir = OUTPUT_BASE_DIR / video_stem
    
    try:
        # Calculate frame requirements
        ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
        tgt_f = PREDICT_TOKENS * TUBELET_SIZE * STRIDE
        total_f_needed = ctx_f + tgt_f
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_v_frames < (total_f_needed + (TUBELET_SIZE * STRIDE)): 
            print(f"Skipping {video_stem}: Too short.")
            cap.release()
            return False

        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Read all frames and preprocess
        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            raw_frames.append(
                cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB)
            )
        cap.release()

        # Extract middle section for analysis
        mid = len(raw_frames) // 2
        tgt_start, tgt_end = mid - (tgt_f // 2), mid + (tgt_f // 2)
        
        # Target frames: the frames we're trying to predict
        target_frames_raw = raw_frames[tgt_start:tgt_end:STRIDE]
        
        # Context frames: past frames used for prediction
        fw_ctx_raw = raw_frames[tgt_start - ctx_f : tgt_start : STRIDE]  # Before target
        bw_ctx_raw = raw_frames[tgt_end : tgt_end + ctx_f : STRIDE][::-1]  # After target (reversed)

        csv_data = {"step": range(CONTEXT_TOKENS + PREDICT_TOKENS)}

        # === BIDIRECTIONAL ANALYSIS ===
        for direction in ["forward", "backward"]:
            print(f"  Processing {direction} direction...")
            
            # Select context and target based on direction
            ctx = fw_ctx_raw if direction == "forward" else bw_ctx_raw
            tgt = target_frames_raw if direction == "forward" else target_frames_raw[::-1]
            
            # ============================================
            # KEY FIX #1: Compute latents separately
            # ============================================
            # Encode context frames ALONE to get context latents
            c_lat = get_vjepa_latents(model, processor, ctx)  # [CONTEXT_TOKENS, P, D]
            
            # Encode target frames ALONE to get ground truth latents
            # (NOT mixed with context - this was the bug!)
            g_lat = get_vjepa_latents(model, processor, tgt)  # [PREDICT_TOKENS, P, D]
            
            # Predict future latents from context only
            p_lat = predict_future_latents(model, c_lat, PREDICT_TOKENS)  # [PREDICT_TOKENS, P, D]
            
            # Compute errors
            cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
            
            # Prep CSV columns (context period has NaN since no prediction happens there)
            full_cos = np.concatenate([np.full(CONTEXT_TOKENS, np.nan), cos_dist])
            full_l2 = np.concatenate([np.full(CONTEXT_TOKENS, np.nan), l2_dist])
            csv_data[f"{direction}_cos"] = full_cos
            csv_data[f"{direction}_l2"] = full_l2

            # ============================================
            # Visualization: Create frame-by-frame GIF
            # ============================================
            full_seq = ctx + tgt  # For display purposes
            viz_frames = []
            
            for t in range(CONTEXT_TOKENS + PREDICT_TOKENS):
                # Get frame image
                frame_idx = min(t * TUBELET_SIZE, len(full_seq) - 1)
                img = Image.fromarray(full_seq[frame_idx])
                
                # Create prediction error plot
                fig, ax1 = plt.subplots(figsize=(6, 4))
                ax1.set_xlabel("Step")
                ax1.set_ylabel("Cosine Distance", color='tab:red')
                ax1.plot(full_cos, color='tab:red', lw=2, label='Cosine')
                ax1.tick_params(axis='y', labelcolor='tab:red')
                ax1.set_ylim(0, 1.0)
                
                ax2 = ax1.twinx()
                ax2.set_ylabel("L2 Distance", color='tab:blue')
                ax2.plot(full_l2, color='tab:blue', lw=2, ls='--', label='L2')
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                ax2.set_ylim(40, 90)
                
                # Mark current prediction point
                if t >= CONTEXT_TOKENS:
                    ax1.scatter(t, full_cos[t], color='tab:red', s=40, zorder=5)
                    ax2.scatter(t, full_l2[t], color='tab:blue', s=40, zorder=5)
                
                # Vertical line separating context from prediction
                ax1.axvline(CONTEXT_TOKENS - 0.5, color='k', linestyle=':', alpha=0.3)
                ax1.set_title(f"{video_stem} | {direction.upper()} | Step {t}")
                fig.tight_layout()
                
                # Convert plot to image
                fig.canvas.draw()
                plot_img = Image.fromarray(np.array(fig.canvas.buffer_rgba())).convert("RGB")
                plt.close(fig)

                # Combine frame and plot
                combined = Image.new('RGB', (384 + plot_img.width, 400), (255, 255, 255))
                combined.paste(img, (0, 8))
                combined.paste(plot_img, (384, 0))
                viz_frames.append(combined)

            # Save animation
            viz_frames[0].save(
                video_dir / f"{direction}_analysis.gif",
                save_all=True,
                append_images=viz_frames[1:],
                duration=150,
                loop=0
            )

        # Save metrics to CSV
        pd.DataFrame(csv_data).to_csv(video_dir / "metrics.csv", index=False)
        
        # Print summary
        f_c = np.nanmean(csv_data['forward_cos'])
        b_c = np.nanmean(csv_data['backward_cos'])
        print(f"[{video_stem}] Mean Cos: FW={f_c:.3f}, BW={b_c:.3f} | Ratio: {b_c/f_c:.2f}")
        return True

    except Exception as e:
        print(f"Error on {video_stem}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
    
    for name in VIDEO_NAMES:
        p = f"/project/3018078.02/MEG_ingmar/shorts/{name}.mp4"
        if os.path.exists(p):
            print(f"Processing {name}...")
            run_analysis(mdl, proc, p)
        else:
            print(f"Video not found: {p}")





# BOTTOM SCRIPT WORKS, BUT LIKELY WORKS WRONG
# IN THE SENSE THAT GROUND TRUTH IS HANDLED POORLY
# AND ALSO THE MASKING IS NOT COMPLETE, AS IN IT PROBABLY STILL HAS
# ACCESS TO GROUND TRUTH INFO





# import os
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pathlib import Path
# from datetime import datetime
# from PIL import Image, ImageDraw
# import torch.nn.functional as F
# from transformers import AutoModel, AutoVideoProcessor

# # --- Configuration ---
# HF_MODEL_NAME     = "facebook/vjepa2-vitg-fpc64-384"
# CONTEXT_TOKENS    = 10   
# PREDICT_TOKENS    = 15   
# TUBELET_SIZE      = 2    
# STRIDE            = 3  
# OUTPUT_BASE_DIR   = Path("vjepa_results_v4")
# # VIDEO_NAMES       = ["deurinhuis", "trap", "titanic_bound", "iglo", "kanon", "sprongwagon", "geiser", 
#                     #  "klepdonder", "gevelstort", "foetsie", "uitzichtdoek", "yenga", "whackamole", 
#                     #  "trein_portret", "gooi_krat", "bw_testclip_pluim", "bw_testclip_lazeren"]
# VIDEO_NAMES      = ["experiment"]  # For quick testing
# import matplotlib
# matplotlib.use('Agg')

# # --- Helper Functions ---

# def calculate_metrics(predicted, ground_truth):
#     p_vec = predicted.mean(dim=1) 
#     g_vec = ground_truth.mean(dim=1) 
#     min_len = min(p_vec.size(0), g_vec.size(0))
#     p_vec, g_vec = p_vec[:min_len], g_vec[:min_len]
    
#     cos_sim = F.cosine_similarity(p_vec, g_vec, dim=-1)
#     cos_dist = (1.0 - cos_sim).numpy()
#     l2_dist = torch.norm(p_vec - g_vec, p=2, dim=-1).numpy()
#     return cos_dist, l2_dist

# @torch.inference_mode()
# def get_vjepa_latents(model, processor, frames):
#     video = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in frames])
#     inputs = processor(video, return_tensors="pt")
#     pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
#     feats = model.get_vision_features(pixel_values).cpu()
#     P = feats.shape[1] // (len(frames) // TUBELET_SIZE)
#     return feats.reshape(-1, P, feats.shape[-1])

# @torch.inference_mode()
# def predict_future_latents(model, context_latents, num_future_steps):
#     T_ctx, P, D = context_latents.shape
#     total_len = T_ctx + num_future_steps
#     predictor = model.predictor
#     device = next(model.parameters()).device
    
#     ctx_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device).to(device)
#     ctx_mask[:, :T_ctx * P] = 1
#     tgt_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device).to(device)
#     tgt_mask[:, T_ctx * P : (T_ctx + num_future_steps) * P] = 1

#     z_ctx = context_latents.reshape(1, -1, D).to(device)
#     z_padded = torch.zeros(1, total_len * P, D, device=device)
#     z_padded[:, :T_ctx * P, :] = z_ctx

#     out = predictor(encoder_hidden_states=z_padded, context_mask=[ctx_mask], target_mask=[tgt_mask])
#     return out.last_hidden_state[:, T_ctx * P :, :].reshape(num_future_steps, P, D).cpu()

# def run_analysis(model, processor, video_path):
#     video_stem = Path(video_path).stem
#     video_dir = OUTPUT_BASE_DIR / video_stem
    
#     try:
#         ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
#         tgt_f = PREDICT_TOKENS * TUBELET_SIZE * STRIDE
#         total_f_needed = ctx_f + tgt_f
        
#         cap = cv2.VideoCapture(video_path)
#         total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if total_v_frames < (total_f_needed + (TUBELET_SIZE * STRIDE)): 
#             print(f"Skipping {video_stem}: Too short.")
#             cap.release()
#             return False

#         video_dir.mkdir(parents=True, exist_ok=True)
#         raw_frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             raw_frames.append(cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB))
#         cap.release()

#         mid = len(raw_frames) // 2
#         tgt_start, tgt_end = mid - (tgt_f // 2), mid + (tgt_f // 2)
#         target_frames_raw = raw_frames[tgt_start:tgt_end:STRIDE]
#         fw_ctx_raw = raw_frames[tgt_start - ctx_f : tgt_start : STRIDE]
#         bw_ctx_raw = raw_frames[tgt_end : tgt_end + ctx_f : STRIDE][::-1]

#         csv_data = {"step": range(CONTEXT_TOKENS + PREDICT_TOKENS)}

#         for direction in ["forward", "backward"]:
#             ctx = fw_ctx_raw if direction == "forward" else bw_ctx_raw
#             tgt = target_frames_raw if direction == "forward" else target_frames_raw[::-1]
#             full_seq = ctx + tgt
#             latents = get_vjepa_latents(model, processor, full_seq)
            
#             c_lat = latents[:CONTEXT_TOKENS]
#             g_lat = latents[CONTEXT_TOKENS : CONTEXT_TOKENS + PREDICT_TOKENS]
#             p_lat = predict_future_latents(model, c_lat, PREDICT_TOKENS)
            
#             cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
            
#             # Prep CSV columns
#             full_cos = np.concatenate([np.full(CONTEXT_TOKENS, np.nan), cos_dist])
#             full_l2 = np.concatenate([np.full(CONTEXT_TOKENS, np.nan), l2_dist])
#             csv_data[f"{direction}_cos"] = full_cos
#             csv_data[f"{direction}_l2"] = full_l2

#             # Visuals
#             viz_frames = []
#             for t in range(CONTEXT_TOKENS + PREDICT_TOKENS):
#                 img = Image.fromarray(full_seq[min(t*TUBELET_SIZE, len(full_seq)-1)])
                
#                 fig, ax1 = plt.subplots(figsize=(6, 4))
#                 ax1.set_xlabel("Step")
#                 ax1.set_ylabel("Cosine Distance", color='tab:red')
#                 ax1.plot(full_cos, color='tab:red', lw=2, label='Cosine')
#                 ax1.tick_params(axis='y', labelcolor='tab:red')
#                 ax1.set_ylim(0, 1.0)
                
#                 ax2 = ax1.twinx()
#                 ax2.set_ylabel("L2 Distance", color='tab:blue')
#                 ax2.plot(full_l2, color='tab:blue', lw=2, ls='--', label='L2')
#                 ax2.tick_params(axis='y', labelcolor='tab:blue')
#                 ax2.set_ylim(40, 90)
                
#                 if t >= CONTEXT_TOKENS:
#                     ax1.scatter(t, full_cos[t], color='tab:red', s=40, zorder=5)
#                     ax2.scatter(t, full_l2[t], color='tab:blue', s=40, zorder=5)
                
#                 ax1.axvline(CONTEXT_TOKENS - 0.5, color='k', linestyle=':', alpha=0.3)
#                 ax1.set_title(f"{video_stem} | {direction.upper()} | Step {t}")
#                 fig.tight_layout()
                
#                 fig.canvas.draw()
#                 plot_img = Image.fromarray(np.array(fig.canvas.buffer_rgba())).convert("RGB")
#                 plt.close(fig)

#                 combined = Image.new('RGB', (384 + plot_img.width, 400), (255, 255, 255))
#                 combined.paste(img, (0, 8))
#                 combined.paste(plot_img, (384, 0))
#                 viz_frames.append(combined)

#             viz_frames[0].save(video_dir/f"{direction}_analysis.gif", save_all=True, append_images=viz_frames[1:], duration=150, loop=0)

#         pd.DataFrame(csv_data).to_csv(video_dir / "metrics.csv", index=False)
        
#         # Printing summary
#         f_c, b_c = np.nanmean(csv_data['forward_cos']), np.nanmean(csv_data['backward_cos'])
#         print(f"[{video_stem}] Mean Cos: FW={f_c:.3f}, BW={b_c:.3f} | Ratio: {b_c/f_c:.2f}")

#     except Exception as e:
#         print(f"Error on {video_stem}: {e}")
#     return True

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
#     mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
#     for name in VIDEO_NAMES:
#         p = f"/project/3018078.02/MEG_ingmar/shorts/{name}.mp4"
#         if os.path.exists(p): run_analysis(mdl, proc, p)
