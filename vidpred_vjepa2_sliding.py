import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor
import matplotlib
matplotlib.use('Agg')

# --- Configuration ---
HF_MODEL_NAME     = "facebook/vjepa2-vitg-fpc64-384"
CONTEXT_TOKENS    = 10   
PREDICT_TOKENS_LIST = [4, 8, 12]  # Different future prediction horizons
TUBELET_SIZE      = 2    
STRIDE            = 2  
OUTPUT_BASE_DIR   = Path("vjepa_results_sliding")
# VIDEO_NAMES       = ["deurinhuis", "trap", "titanic_bound", "iglo", "kanon", "sprongwagon", "geiser", 
                    #  "klepdonder", "gevelstort", "foetsie", "uitzichtdoek", "yenga", "whackamole", 
                    #  "trein_portret", "gooi_krat", "bw_testclip_pluim", "bw_testclip_lazeren"]
# VIDEO_NAMES = ["ProjectAttention_movie_part21_24Hz"] # one longer one
VIDEO_NAMES = ["longclip_part42_bw"] # one longer one

# --- Visual Statistics Functions ---

def compute_rms_contrast(frame):
    """Compute RMS (root mean square) contrast of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.sqrt(np.mean(gray ** 2))

def compute_brightness(frame):
    """Compute mean brightness of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.mean(gray)

def compute_edge_content(frame):
    """Compute edge density using Canny edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.mean(edges) / 255.0

def compute_optical_flow_magnitude(frame1, frame2):
    """Compute mean optical flow magnitude between two consecutive frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(magnitude)

def compute_temporal_variance(frames_list):
    """Compute temporal variance across a sequence of frames."""
    if len(frames_list) < 2:
        return 0.0
    
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for f in frames_list]
    temporal_var = np.var([np.mean(g) for g in grays])
    return temporal_var

def compute_visual_statistics(frame, prev_frame=None):
    """Compute all visual statistics for a frame."""
    stats = {
        'rms_contrast': compute_rms_contrast(frame),
        'brightness': compute_brightness(frame),
        'edge_content': compute_edge_content(frame),
    }
    
    if prev_frame is not None:
        stats['optical_flow_magnitude'] = compute_optical_flow_magnitude(prev_frame, frame)
    else:
        stats['optical_flow_magnitude'] = np.nan
    
    return stats

# --- Helper Functions ---

def calculate_metrics(predicted, ground_truth):
    """Calculate cosine and L2 distances between predicted and ground truth latents."""
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
    """Extract V-JEPA latent features from a sequence of frames."""
    video = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in frames])
    inputs = processor(video, return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
    feats = model.get_vision_features(pixel_values).cpu()
    P = feats.shape[1] // (len(frames) // TUBELET_SIZE)
    return feats.reshape(-1, P, feats.shape[-1])

@torch.inference_mode()
def predict_future_latents(model, context_latents, num_future_steps):
    """Predict future latents given context latents."""
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

def run_sliding_analysis(model, processor, video_path):
    """Run sliding window analysis on video, computing predictability and visual stats."""
    video_stem = Path(video_path).stem
    video_dir = OUTPUT_BASE_DIR / video_stem
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {video_stem}")
        print(f"{'='*60}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if video is long enough
        ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
        min_frames_needed = ctx_f + max(PREDICT_TOKENS_LIST) * TUBELET_SIZE * STRIDE
        
        if total_v_frames < min_frames_needed:
            print(f"Skipping {video_stem}: Video too short ({total_v_frames} < {min_frames_needed} frames)")
            cap.release()
            return False
        
        # Create output directory
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all frames
        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB))
        cap.release()
        
        print(f"Loaded {len(raw_frames)} frames")
        
        # Prepare output data structure
        frame_results = []
        max_predict = max(PREDICT_TOKENS_LIST)
        
        # Sliding window analysis
        num_windows = len(raw_frames) - ctx_f - (max_predict * TUBELET_SIZE * STRIDE)
        
        print(f"Running sliding window analysis ({num_windows} windows)...")
        
        for window_idx in range(num_windows):
            if (window_idx + 1) % max(1, num_windows // 10) == 0:
                print(f"  Progress: {window_idx + 1}/{num_windows} windows processed")
            
            # Get context frames
            ctx_start = window_idx
            ctx_end = ctx_start + ctx_f
            context_frames_raw = raw_frames[ctx_start:ctx_end:STRIDE]
            
            # Get ground truth frames (future frames to predict)
            gt_start = ctx_end
            max_gt_end = gt_start + (max_predict * TUBELET_SIZE * STRIDE)
            all_future_frames_raw = raw_frames[gt_start:max_gt_end:STRIDE]
            
            # Extract V-JEPA latents for context + all future frames
            full_seq = context_frames_raw + all_future_frames_raw
            
            try:
                latents = get_vjepa_latents(model, processor, full_seq)
            except Exception as e:
                print(f"    Warning: Latent extraction failed at window {window_idx}: {e}")
                continue
            
            c_lat = latents[:CONTEXT_TOKENS]
            
            # Compute visual statistics for context frames
            vis_stats = {
                'frame_index': window_idx,
            }
            
            # Context visual stats
            for stat_name in ['rms_contrast', 'brightness', 'edge_content', 'optical_flow_magnitude']:
                vis_stats[f'ctx_{stat_name}'] = []
            
            prev_frame = None
            for i, frame in enumerate(context_frames_raw):
                stats = compute_visual_statistics(frame, prev_frame)
                for stat_name, stat_val in stats.items():
                    vis_stats[f'ctx_{stat_name}'].append(stat_val)
                prev_frame = frame
            
            # Take mean of context stats
            for stat_name in ['rms_contrast', 'brightness', 'edge_content', 'optical_flow_magnitude']:
                key = f'ctx_{stat_name}'
                if vis_stats[key]:
                    vis_stats[key] = np.nanmean(vis_stats[key])
                else:
                    vis_stats[key] = np.nan
            
            # Compute predictions for different future horizons
            for predict_steps in PREDICT_TOKENS_LIST:
                g_lat = latents[CONTEXT_TOKENS:CONTEXT_TOKENS + predict_steps]
                
                if len(g_lat) < predict_steps:
                    # Pad if needed
                    padding = predict_steps - len(g_lat)
                    g_lat = F.pad(g_lat, (0, 0, 0, 0, 0, padding))
                
                # Predict future latents
                p_lat = predict_future_latents(model, c_lat, predict_steps)
                
                # Calculate metrics
                cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
                
                # Store mean metrics
                vis_stats[f'pred_{predict_steps}_cos_mean'] = np.mean(cos_dist)
                vis_stats[f'pred_{predict_steps}_l2_mean'] = np.mean(l2_dist)
                vis_stats[f'pred_{predict_steps}_cos_std'] = np.std(cos_dist)
                vis_stats[f'pred_{predict_steps}_l2_std'] = np.std(l2_dist)
            
            frame_results.append(vis_stats)
        
        # Save results to CSV
        if frame_results:
            df = pd.DataFrame(frame_results)
            csv_path = video_dir / "frame_metrics.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
            
            # Print summary statistics
            print(f"\nSummary Statistics for {video_stem}:")
            print(f"  Frames analyzed: {len(frame_results)}")
            print(f"  Context RMS Contrast: {df['ctx_rms_contrast'].mean():.4f} ± {df['ctx_rms_contrast'].std():.4f}")
            print(f"  Context Brightness: {df['ctx_brightness'].mean():.4f} ± {df['ctx_brightness'].std():.4f}")
            print(f"  Context Edge Content: {df['ctx_edge_content'].mean():.4f} ± {df['ctx_edge_content'].std():.4f}")
            print(f"  Context Optical Flow: {df['ctx_optical_flow_magnitude'].mean():.4f} ± {df['ctx_optical_flow_magnitude'].std():.4f}")
            
            for pred_steps in PREDICT_TOKENS_LIST:
                cos_col = f'pred_{pred_steps}_cos_mean'
                l2_col = f'pred_{pred_steps}_l2_mean'
                print(f"\n  Prediction {pred_steps} steps:")
                print(f"    Cosine Distance: {df[cos_col].mean():.4f} ± {df[cos_col].std():.4f}")
                print(f"    L2 Distance: {df[l2_col].mean():.4f} ± {df[l2_col].std():.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error processing {video_stem}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function."""
    print(f"\nStarting V-JEPA Sliding Window Analysis")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {HF_MODEL_NAME}")
    proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
    print("Model loaded successfully")
    
    # Create output directory
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process videos
    processed_count = 0
    failed_count = 0
    
    for video_name in VIDEO_NAMES:
        video_path = f"/project/3018078.02/MEG_ingmar/shorts/{video_name}.mp4"
        # video_path = f"/project/3018078.02/MEG_ingmar/{video_name}.mp4"
        
        if os.path.exists(video_path):
            success = run_sliding_analysis(mdl, proc, video_path)
            if success:
                processed_count += 1
            else:
                failed_count += 1
        else:
            print(f"Video not found: {video_path}")
            failed_count += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"Successfully processed: {processed_count} videos")
    print(f"Failed: {failed_count} videos")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Results in: {OUTPUT_BASE_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
