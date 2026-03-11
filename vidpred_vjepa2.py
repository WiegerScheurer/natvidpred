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
STRIDE            = 2  
OUTPUT_BASE_DIR   = Path("vjepa_results_v4")
VIDEO_NAMES       = ["deurinhuis", "trap", "titanic_bound", "iglo", "kanon", "sprongwagon", "geiser", 
                     "klepdonder", "gevelstort", "foetsie", "uitzichtdoek", "yenga", "whackamole", 
                     "trein_portret", "gooi_krat", "bw_testclip_pluim", "bw_testclip_lazeren"]

import matplotlib
matplotlib.use('Agg')

# --- Helper Functions ---

def calculate_metrics(predicted, ground_truth):
    p_vec = predicted.mean(dim=1) 
    g_vec = ground_truth.mean(dim=1) 
    min_len = min(p_vec.size(0), g_vec.size(0))
    p_vec, g_vec = p_vec[:min_len], g_vec[:min_len]
    
    cos_sim = F.cosine_similarity(p_vec, g_vec, dim=-1)
    cos_dist = (1.0 - cos_sim).numpy()
    l2_dist = torch.norm(p_vec - g_vec, p=2, dim=-1).numpy()
    return cos_dist, l2_dist

@torch.inference_mode()
def get_vjepa_latents(model, processor, frames):
    video = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in frames])
    inputs = processor(video, return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
    feats = model.get_vision_features(pixel_values).cpu()
    P = feats.shape[1] // (len(frames) // TUBELET_SIZE)
    return feats.reshape(-1, P, feats.shape[-1])

@torch.inference_mode()
def predict_future_latents(model, context_latents, num_future_steps):
    T_ctx, P, D = context_latents.shape
    total_len = T_ctx + num_future_steps
    predictor = model.predictor
    device = next(model.parameters()).device
    
    ctx_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device).to(device)
    ctx_mask[:, :T_ctx * P] = 1
    tgt_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device).to(device)
    tgt_mask[:, T_ctx * P : (T_ctx + num_future_steps) * P] = 1

    z_ctx = context_latents.reshape(1, -1, D).to(device)
    z_padded = torch.zeros(1, total_len * P, D, device=device)
    z_padded[:, :T_ctx * P, :] = z_ctx

    out = predictor(encoder_hidden_states=z_padded, context_mask=[ctx_mask], target_mask=[tgt_mask])
    return out.last_hidden_state[:, T_ctx * P :, :].reshape(num_future_steps, P, D).cpu()

def run_analysis(model, processor, video_path):
    video_stem = Path(video_path).stem
    video_dir = OUTPUT_BASE_DIR / video_stem
    
    try:
        ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
        tgt_f = PREDICT_TOKENS * TUBELET_SIZE * STRIDE
        total_f_needed = ctx_f + tgt_f
        
        cap = cv2.VideoCapture(video_path)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_v_frames < (total_f_needed + (TUBELET_SIZE * STRIDE)): 
            print(f"Skipping {video_stem}: Too short.")
            cap.release()
            return False

        video_dir.mkdir(parents=True, exist_ok=True)
        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            raw_frames.append(cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB))
        cap.release()

        mid = len(raw_frames) // 2
        tgt_start, tgt_end = mid - (tgt_f // 2), mid + (tgt_f // 2)
        target_frames_raw = raw_frames[tgt_start:tgt_end:STRIDE]
        fw_ctx_raw = raw_frames[tgt_start - ctx_f : tgt_start : STRIDE]
        bw_ctx_raw = raw_frames[tgt_end : tgt_end + ctx_f : STRIDE][::-1]

        csv_data = {"step": range(CONTEXT_TOKENS + PREDICT_TOKENS)}

        for direction in ["forward", "backward"]:
            ctx = fw_ctx_raw if direction == "forward" else bw_ctx_raw
            tgt = target_frames_raw if direction == "forward" else target_frames_raw[::-1]
            full_seq = ctx + tgt
            latents = get_vjepa_latents(model, processor, full_seq)
            
            c_lat = latents[:CONTEXT_TOKENS]
            g_lat = latents[CONTEXT_TOKENS : CONTEXT_TOKENS + PREDICT_TOKENS]
            p_lat = predict_future_latents(model, c_lat, PREDICT_TOKENS)
            
            cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
            
            # Prep CSV columns
            full_cos = np.concatenate([np.full(CONTEXT_TOKENS, np.nan), cos_dist])
            full_l2 = np.concatenate([np.full(CONTEXT_TOKENS, np.nan), l2_dist])
            csv_data[f"{direction}_cos"] = full_cos
            csv_data[f"{direction}_l2"] = full_l2

            # Visuals
            viz_frames = []
            for t in range(CONTEXT_TOKENS + PREDICT_TOKENS):
                img = Image.fromarray(full_seq[min(t*TUBELET_SIZE, len(full_seq)-1)])
                
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
                
                if t >= CONTEXT_TOKENS:
                    ax1.scatter(t, full_cos[t], color='tab:red', s=40, zorder=5)
                    ax2.scatter(t, full_l2[t], color='tab:blue', s=40, zorder=5)
                
                ax1.axvline(CONTEXT_TOKENS - 0.5, color='k', linestyle=':', alpha=0.3)
                ax1.set_title(f"{video_stem} | {direction.upper()} | Step {t}")
                fig.tight_layout()
                
                fig.canvas.draw()
                plot_img = Image.fromarray(np.array(fig.canvas.buffer_rgba())).convert("RGB")
                plt.close(fig)

                combined = Image.new('RGB', (384 + plot_img.width, 400), (255, 255, 255))
                combined.paste(img, (0, 8))
                combined.paste(plot_img, (384, 0))
                viz_frames.append(combined)

            viz_frames[0].save(video_dir/f"{direction}_analysis.gif", save_all=True, append_images=viz_frames[1:], duration=150, loop=0)

        pd.DataFrame(csv_data).to_csv(video_dir / "metrics.csv", index=False)
        
        # Printing summary
        f_c, b_c = np.nanmean(csv_data['forward_cos']), np.nanmean(csv_data['backward_cos'])
        print(f"[{video_stem}] Mean Cos: FW={f_c:.3f}, BW={b_c:.3f} | Ratio: {b_c/f_c:.2f}")

    except Exception as e:
        print(f"Error on {video_stem}: {e}")
    return True

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
    for name in VIDEO_NAMES:
        p = f"/project/3018078.02/MEG_ingmar/shorts/{name}.mp4"
        if os.path.exists(p): run_analysis(mdl, proc, p)










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
# STRIDE            = 1  
# OUTPUT_BASE_DIR   = Path("vjepa_results_v3")
# VIDEO_NAMES       = ["deurinhuis", "trap", "titanic_bound", "iglo", "kanon", "sprongwagon", "geiser", 
#                      "klepdonder", "gevelstort", "foetsie", "uitzichtdoek", "yenga", "whackamole", 
#                      "trein_portret", "gooi_krat", "bw_testclip_pluim", "bw_testclip_lazeren"]

# import matplotlib
# matplotlib.use('Agg')

# # --- Helper Functions ---

# def calculate_metrics(predicted, ground_truth):
#     p_vec = predicted.mean(dim=1) 
#     g_vec = ground_truth.mean(dim=1) 
    
#     # Ensure lengths match before math
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
#         # 1. Setup
#         ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
#         tgt_f = PREDICT_TOKENS * TUBELET_SIZE * STRIDE
#         total_f_needed = ctx_f + tgt_f
        
#         cap = cv2.VideoCapture(video_path)
#         total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         if total_v_frames < (total_f_needed + (TUBELET_SIZE * STRIDE)): 
#             print(f"Skipping {video_stem}: Need ~{total_f_needed} frames, found {total_v_frames}")
#             cap.release()
#             return False

#         video_dir.mkdir(parents=True, exist_ok=True)
#         raw_frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             raw_frames.append(cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB))
#         cap.release()

#         # 2. Slice Logic
#         mid = len(raw_frames) // 2
#         tgt_start, tgt_end = mid - (tgt_f // 2), mid + (tgt_f // 2)
        
#         # Validation of slice boundaries
#         if tgt_start - ctx_f < 0 or tgt_end + ctx_f > len(raw_frames):
#             print(f"Skipping {video_stem}: Center-crop out of bounds.")
#             return False

#         target_frames_raw = raw_frames[tgt_start:tgt_end:STRIDE]
#         fw_ctx_raw = raw_frames[tgt_start - ctx_f : tgt_start : STRIDE]
#         bw_ctx_raw = raw_frames[tgt_end : tgt_end + ctx_f : STRIDE][::-1]

#         # 3. Process Directions
#         dir_results = {}
#         for direction in ["forward", "backward"]:
#             ctx = fw_ctx_raw if direction == "forward" else bw_ctx_raw
#             tgt = target_frames_raw if direction == "forward" else target_frames_raw[::-1]
            
#             full_seq = ctx + tgt
#             latents = get_vjepa_latents(model, processor, full_seq)
            
#             # Robust slicing of latents
#             c_lat = latents[:CONTEXT_TOKENS]
#             g_lat = latents[CONTEXT_TOKENS : CONTEXT_TOKENS + PREDICT_TOKENS]
            
#             p_lat = predict_future_latents(model, c_lat, PREDICT_TOKENS)
#             cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
            
#             dir_results[direction] = {
#                 'cos': cos_dist, 
#                 'l2': l2_dist, 
#                 'seq': full_seq,
#                 'mean_cos': cos_dist.mean(),
#                 'mean_l2': l2_dist.mean()
#             }

#         # 4. Final Comparison & Visualization
#         for direction in ["forward", "backward"]:
#             viz_frames = []
#             f_cos = dir_results['forward']['cos']
#             b_cos = dir_results['backward']['cos']
#             current_cos = dir_results[direction]['cos']
#             current_l2 = dir_results[direction]['l2']

#             for t in range(CONTEXT_TOKENS + PREDICT_TOKENS):
#                 img = Image.fromarray(dir_results[direction]['seq'][min(t*TUBELET_SIZE, len(target_frames_raw)*2-1)])
                
#                 fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
                
#                 # Top Plot: Individual metrics for this direction
#                 ax1.plot(range(CONTEXT_TOKENS, CONTEXT_TOKENS+PREDICT_TOKENS), current_cos, color='red', label='Cosine')
#                 ax1.set_ylim(0, 1)
#                 ax1_twin = ax1.twinx()
#                 ax1_twin.plot(range(CONTEXT_TOKENS, CONTEXT_TOKENS+PREDICT_TOKENS), current_l2, color='blue', ls='--', label='L2')
#                 ax1_twin.set_ylim(40, 85)
#                 ax1.set_title(f"{direction.upper()} Metrics")
                
#                 # Bottom Plot: The "Battle" (Forward vs Backward Cosine)
#                 ax2.plot(range(CONTEXT_TOKENS, CONTEXT_TOKENS+PREDICT_TOKENS), f_cos, color='green', alpha=0.6, label='Forward')
#                 ax2.plot(range(CONTEXT_TOKENS, CONTEXT_TOKENS+PREDICT_TOKENS), b_cos, color='purple', alpha=0.6, label='Backward')
#                 ax2.set_ylim(0, 1)
#                 ax2.set_title("Direct Comparison (Cosine)")
#                 ax2.legend(loc='upper left', fontsize='small')
                
#                 # Tracking dots
#                 if t >= CONTEXT_TOKENS:
#                     idx = t - CONTEXT_TOKENS
#                     ax1.scatter(t, current_cos[idx], color='red')
#                     ax2.scatter(t, f_cos[idx] if direction=='forward' else b_cos[idx], color='black')

#                 plt.tight_layout()
#                 fig.canvas.draw()
#                 plot_img = Image.fromarray(np.array(fig.canvas.buffer_rgba())).convert("RGB")
#                 plt.close(fig)

#                 combined = Image.new('RGB', (384 + plot_img.width, plot_img.height), (255,255,255))
#                 combined.paste(img, (0, (plot_img.height-384)//2))
#                 combined.paste(plot_img, (384, 0))
#                 viz_frames.append(combined)

#             viz_frames[0].save(video_dir/f"{direction}_comparison.gif", save_all=True, append_images=viz_frames[1:], duration=150, loop=0)

#         # Print Final Results
#         print(f"\n--- {video_stem} ---")
#         print(f"COS: FW={dir_results['forward']['mean_cos']:.3f} | BW={dir_results['backward']['mean_cos']:.3f} | Ratio={dir_results['backward']['mean_cos']/dir_results['forward']['mean_cos']:.2f}")
#         print(f"L2:  FW={dir_results['forward']['mean_l2']:.1f} | BW={dir_results['backward']['mean_l2']:.1f} | Ratio={dir_results['backward']['mean_l2']/dir_results['forward']['mean_l2']:.2f}")

#     except Exception as e:
#         print(f"ERROR processing {video_stem}: {e}")
#     return True

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
#     mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
#     for name in VIDEO_NAMES:
#         p = f"/project/3018078.02/MEG_ingmar/shorts/{name}.mp4"
#         if os.path.exists(p): run_analysis(mdl, proc, p)











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
# PREDICT_TOKENS    = 15   # TOTAL_STEPS was 25 (10 context + 15 prediction)
# TUBELET_SIZE      = 2    
# STRIDE            = 2  
# OUTPUT_BASE_DIR   = Path("vjepa_results_v3")
# # VIDEO_NAMES       = ["deurinhuis", "trap", "geiser", "kanon"]
# VIDEO_NAMES = ["deurinhuis", "trap", "titanic_bound",
#                 "iglo", "kanon", "sprongwagon", "geiser", 
#                 "klepdonder", "gevelstort", "foetsie", "uitzichtdoek", "yenga", "whackamole", 
#                 "uitzichtdoek", "foetsie", "gevelstort", "trein_portret", "gooi_krat", 
#                 "bw_testclip_pluim", "bw_testclip_lazeren", "kanon"]

# import matplotlib
# matplotlib.use('Agg')

# # --- Helper Functions ---

# def calculate_metrics(predicted, ground_truth):
#     p_vec = predicted.mean(dim=1) 
#     g_vec = ground_truth.mean(dim=1) 
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
#     video_dir.mkdir(parents=True, exist_ok=True)
    
#     # Calculate frame requirements
#     ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
#     tgt_f = PREDICT_TOKENS * TUBELET_SIZE * STRIDE
#     total_f_needed = ctx_f + tgt_f
    
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     if total_v_frames < total_f_needed:
#         print(f"Skipping {video_stem}: Needs {total_f_needed} frames, has {total_v_frames}")
#         cap.release()
#         return False

#     # Extract ALL frames once
#     raw_frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         raw_frames.append(cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB))
#     cap.release()

#     # --- CENTER TARGET ALIGNMENT ---
#     # Pick the exact middle of the video for the target
#     mid_point = len(raw_frames) // 2
#     tgt_start = mid_point - (tgt_f // 2)
#     tgt_end = tgt_start + tgt_f
    
#     # Slice the target frames (Global Ground Truth)
#     target_frames_raw = raw_frames[tgt_start:tgt_end:STRIDE]
    
#     # Slice Contexts
#     fw_context_raw = raw_frames[tgt_start - ctx_f : tgt_start : STRIDE]
#     bw_context_raw = raw_frames[tgt_end : tgt_end + ctx_f : STRIDE][::-1] # Reverse of trailing frames

#     results_summary = {}

#     for direction in ["forward", "backward"]:
#         ctx = fw_context_raw if direction == "forward" else bw_context_raw
#         # The target must be reversed for the backward model to "predict" it
#         tgt = target_frames_raw if direction == "forward" else target_frames_raw[::-1]
        
#         # Combine for encoding
#         full_seq = ctx + tgt
#         latents = get_vjepa_latents(model, processor, full_seq)
        
#         ctx_latents = latents[:CONTEXT_TOKENS]
#         gt_target_latents = latents[CONTEXT_TOKENS:]
        
#         pred_latents = predict_future_latents(model, ctx_latents, PREDICT_TOKENS)
#         cos_dist, l2_dist = calculate_metrics(pred_latents, gt_target_latents)
        
#         # Visualization (Restored Tracking Dot & Twin Axis)
#         full_cos = np.concatenate([np.full(CONTEXT_TOKENS, np.nan), cos_dist])
#         full_l2 = np.concatenate([np.full(CONTEXT_TOKENS, np.nan), l2_dist])
        
#         viz_frames = []
#         for t in range(CONTEXT_TOKENS + PREDICT_TOKENS):
#             img_idx = min(t * TUBELET_SIZE, len(full_seq)-1)
#             img = Image.fromarray(full_seq[img_idx])
            
#             fig, ax1 = plt.subplots(figsize=(6, 4))
#             ax1.plot(full_cos, color='tab:red', lw=2)
#             ax1.set_ylim(0, 1.0)
#             ax1.set_ylabel("Cosine Dist", color='tab:red')
            
#             ax2 = ax1.twinx()
#             ax2.plot(full_l2, color='tab:blue', lw=2, ls='--')
#             ax2.set_ylim(40, 85)
#             ax2.set_ylabel("L2 Dist", color='tab:blue')
            
#             if not np.isnan(full_cos[t]):
#                 ax1.scatter(t, full_cos[t], color='tab:red', s=50, zorder=5)
#                 ax2.scatter(t, full_l2[t], color='tab:blue', s=50, zorder=5)
            
#             ax1.axvline(CONTEXT_TOKENS-0.5, color='k', alpha=0.2)
#             plt.title(f"{video_stem} | {direction}")
#             fig.tight_layout()
            
#             fig.canvas.draw()
#             plot_img = Image.fromarray(np.array(fig.canvas.buffer_rgba())).convert("RGB")
#             plt.close(fig)

#             combined = Image.new('RGB', (384 + plot_img.width, 400), (255,255,255))
#             combined.paste(img, (0, 8))
#             combined.paste(plot_img, (384, 0))
#             viz_frames.append(combined)

#         gif_name = f"{direction}_stride{STRIDE}.gif"
#         viz_frames[0].save(video_dir/gif_name, save_all=True, append_images=viz_frames[1:], duration=150, loop=0)
        
#         results_summary[direction] = {'cos': cos_dist.mean(), 'l2': l2_dist.mean()}

#     # --- PRINT DISCREPANCY METRICS ---
#     f_cos, b_cos = results_summary['forward']['cos'], results_summary['backward']['cos']
#     f_l2, b_l2   = results_summary['forward']['l2'], results_summary['backward']['l2']
    
#     print(f"\n--- Results for: {video_stem} ---")
#     print(f"COSINE: Forward={f_cos:.3f} | Backward={b_cos:.3f} | Ratio={b_cos/f_cos:.2f}")
#     print(f"L2 DIST: Forward={f_l2:.1f} | Backward={b_l2:.1f} | Ratio={b_l2/f_l2:.2f}")
#     print("-" * 30)
    
#     return True

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
#     mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()

#     for name in VIDEO_NAMES:
#         p = f"/project/3018078.02/MEG_ingmar/shorts/{name}.mp4"
#         if os.path.exists(p): run_analysis(mdl, proc, p)



