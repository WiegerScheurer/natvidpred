"""
V-JEPA2 Sliding Window Predictability Analysis (v3 - Token-Aware)
==================================================================

IMPROVEMENTS in v3:
1. Explicit awareness of 16-token architecture limit
2. Automatic validation of token budget
3. Clear warnings when truncation may occur
4. Option to auto-clamp predictions to available token budget

WHAT THIS SCRIPT DOES:
1. Slides a moving window through the entire video
2. For each window position, extracts:
   - Context frames: the past N frames (used for prediction)
   - Target frames: future M frames (what we're trying to predict)
3. Encodes context and target frames independently
4. Predicts future latents from context only
5. Measures prediction error at different future horizons
6. Correlates prediction error with visual statistics

KEY INSIGHT:
By sliding the window, we get frame-by-frame estimates of "predictability"
- Frames with high motion/change → higher prediction error
- Frames with stable content → lower prediction error

USAGE:
  python vjepa2_sliding_window_v3.py --video foetsie --order forward
  python vjepa2_sliding_window_v3.py --video foetsie --order backward --no-auto-clamp
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor
import matplotlib
matplotlib.use('Agg')

# --- Configuration ---
HF_MODEL_NAME     = "facebook/vjepa2-vitg-fpc64-384"
CONTEXT_TOKENS    = 8   
PREDICT_TOKENS_LIST = [2, 4, 8]  # Different future prediction horizons
TUBELET_SIZE      = 2    
STRIDE            = 2  
OUTPUT_BASE_DIR   = Path("vjepa_results_sliding")
VIDEO_PATH_TEMPLATE = "/project/3018078.02/MEG_ingmar/shorts/{video_name}.mp4"

# Architecture constraint
MAX_TOTAL_TOKENS = 16  # V-JEPA2 sequence length limit
WARN_IF_EXCEEDS_TOKENS = True

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

# --- Token Budget Management ---

def check_token_budget(context_tokens, predict_tokens_list, auto_clamp=True):
    """
    Check if prediction requests fit within token budget.
    
    Args:
        context_tokens: Number of context tokens
        predict_tokens_list: List of requested prediction tokens
        auto_clamp: If True, automatically limit predictions to available budget
    
    Returns:
        valid_predict_tokens: List of prediction tokens that fit (or clamped)
        warnings: List of warning messages
    """
    warnings = []
    available_pred_tokens = MAX_TOTAL_TOKENS - context_tokens
    
    warnings.append(f"Token Budget Analysis:")
    warnings.append(f"  Total token limit: {MAX_TOTAL_TOKENS}")
    warnings.append(f"  Context tokens: {context_tokens}")
    warnings.append(f"  Available for prediction: {available_pred_tokens}")
    
    valid_predict_tokens = []
    
    for pred_tokens in predict_tokens_list:
        if pred_tokens <= available_pred_tokens:
            valid_predict_tokens.append(pred_tokens)
            status = "✓"
        else:
            status = "⚠️"
            if auto_clamp:
                valid_predict_tokens.append(available_pred_tokens)
                warnings.append(
                    f"  {status} Requested {pred_tokens} tokens → "
                    f"CLAMPED to {available_pred_tokens} (available budget)"
                )
            else:
                warnings.append(
                    f"  {status} Requested {pred_tokens} tokens → "
                    f"EXCEEDS budget (only {available_pred_tokens} available). "
                    f"May be TRUNCATED by model."
                )
    
    return valid_predict_tokens, warnings

def apply_frame_order(frames, order):
    """
    Apply frame ordering transformation.
    
    Args:
        frames: List of frame arrays
        order: 'forward', 'backward', or 'shuffled'
    
    Returns:
        Transformed frames list
    """
    if order == 'forward':
        return frames
    elif order == 'backward':
        return frames[::-1]
    elif order == 'shuffled':
        shuffled = frames.copy()
        np.random.shuffle(shuffled)
        return shuffled
    else:
        raise ValueError(f"Unknown frame order: {order}")

def calculate_metrics(predicted, ground_truth):
    """
    Calculate cosine and L2 distances between predicted and ground truth latents.
    
    Args:
        predicted: [T, P, D] predicted latents
        ground_truth: [T, P, D] ground truth latents
    
    Returns:
        cos_dist: cosine distance per timestep
        l2_dist: L2 distance per timestep
    """
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
    """
    Extract V-JEPA latent features from a sequence of frames.
    
    Args:
        model: V-JEPA2 model
        processor: V-JEPA2 video processor
        frames: List of numpy arrays [H, W, 3] in RGB
    
    Returns:
        feats: [T_out, P, D] latent features
    """
    video = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames])
    inputs = processor(video, return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
    feats = model.get_vision_features(pixel_values).cpu()
    P = feats.shape[1] // (len(frames) // TUBELET_SIZE)
    return feats.reshape(-1, P, feats.shape[-1])

@torch.inference_mode()
def predict_future_latents(model, context_latents, num_future_steps):
    """
    Predict future latents given context latents.
    
    CRITICAL: Only context latents are visible to the predictor (causal prediction).
    
    Args:
        model: V-JEPA2 model with predictor
        context_latents: [T_ctx, P, D] context latents only
        num_future_steps: Number of future steps to predict
    
    Returns:
        predicted_latents: [num_future_steps, P, D] predicted future latents
    """
    T_ctx, P, D = context_latents.shape
    total_len = T_ctx + num_future_steps
    predictor = model.predictor
    device = next(model.parameters()).device
    
    # Context mask: 1 for positions with real context, 0 for future (hidden)
    context_mask = torch.ones(1, T_ctx * P, dtype=torch.int64, device=device)
    context_mask = torch.cat([
        context_mask,
        torch.zeros(1, num_future_steps * P, dtype=torch.int64, device=device)
    ], dim=1)
    
    # Target mask: 1 for positions to predict
    target_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
    target_mask[:, T_ctx * P : (T_ctx + num_future_steps) * P] = 1
    
    # Build padded sequence: [context latents | zeros for future]
    z_padded = torch.zeros(1, total_len * P, D, device=device)
    z_ctx = context_latents.reshape(1, -1, D).to(device)
    z_padded[:, :T_ctx * P, :] = z_ctx
    
    # Run predictor
    out = predictor(
        encoder_hidden_states=z_padded,
        context_mask=[context_mask],
        target_mask=[target_mask]
    )
    
    return out.last_hidden_state[:, T_ctx * P :, :].reshape(num_future_steps, P, D).cpu()

def run_sliding_analysis(model, processor, video_path, frame_order='forward', predict_tokens_list=None):
    """
    Run sliding window analysis on video, computing predictability and visual stats.
    
    Args:
        model: V-JEPA2 model
        processor: V-JEPA2 processor
        video_path: Path to video file
        frame_order: 'forward', 'backward', or 'shuffled'
        predict_tokens_list: Override prediction token list
    """
    if predict_tokens_list is None:
        predict_tokens_list = PREDICT_TOKENS_LIST
    
    video_stem = Path(video_path).stem
    video_dir = OUTPUT_BASE_DIR / video_stem
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {video_stem} (order: {frame_order})")
        print(f"{'='*60}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if video is long enough
        ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
        max_predict = max(predict_tokens_list)
        min_frames_needed = ctx_f + (max_predict * TUBELET_SIZE * STRIDE)
        
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
        
        # Apply frame order transformation
        raw_frames = apply_frame_order(raw_frames, frame_order)
        print(f"Applied frame order: {frame_order}")
        
        # Prepare output data structure
        frame_results = []
        
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
            
            try:
                # Encode context frames ALONE
                c_lat = get_vjepa_latents(model, processor, context_frames_raw)
            except Exception as e:
                print(f"    Warning: Context latent extraction failed at window {window_idx}: {e}")
                continue
            
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
            for predict_tokens in predict_tokens_list:
                predict_frames = predict_tokens * TUBELET_SIZE
                
                # Encode ground truth for THIS horizon only
                try:
                    gt_frames_for_horizon = all_future_frames_raw[:predict_frames]
                    g_lat = get_vjepa_latents(model, processor, gt_frames_for_horizon)
                except Exception as e:
                    print(f"    Warning: Ground truth encoding failed for {predict_frames}-frame horizon: {e}")
                    vis_stats[f'pred_{predict_tokens}_cos_mean'] = np.nan
                    vis_stats[f'pred_{predict_tokens}_l2_mean'] = np.nan
                    vis_stats[f'pred_{predict_tokens}_cos_std'] = np.nan
                    vis_stats[f'pred_{predict_tokens}_l2_std'] = np.nan
                    vis_stats[f'pred_{predict_tokens}_cos_max'] = np.nan
                    vis_stats[f'pred_{predict_tokens}_l2_max'] = np.nan
                    for t in range(predict_frames):
                        vis_stats[f'pred_{predict_tokens}_l2_frame_{t}'] = np.nan
                    continue
                
                # Predict future latents from context only
                p_lat = predict_future_latents(model, c_lat, predict_tokens)
                
                # Calculate metrics
                cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
                
                # Store metrics: mean, std, and max
                vis_stats[f'pred_{predict_tokens}_cos_mean'] = np.mean(cos_dist)
                vis_stats[f'pred_{predict_tokens}_l2_mean'] = np.mean(l2_dist)
                vis_stats[f'pred_{predict_tokens}_cos_std'] = np.std(cos_dist)
                vis_stats[f'pred_{predict_tokens}_l2_std'] = np.std(l2_dist)
                vis_stats[f'pred_{predict_tokens}_cos_max'] = np.max(cos_dist)
                vis_stats[f'pred_{predict_tokens}_l2_max'] = np.max(l2_dist)
                
                # Per-frame error
                for t in range(len(l2_dist)):
                    vis_stats[f'pred_{predict_tokens}_l2_frame_{t}'] = l2_dist[t]
            
            frame_results.append(vis_stats)
        
        # Save results to CSV
        if frame_results:
            df = pd.DataFrame(frame_results)
            csv_path = video_dir / f"frame_metrics_{frame_order}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
            
            # Print summary statistics
            print(f"\nSummary Statistics for {video_stem} ({frame_order}):")
            print(f"  Frames analyzed: {len(frame_results)}")
            print(f"  Context RMS Contrast: {df['ctx_rms_contrast'].mean():.4f} ± {df['ctx_rms_contrast'].std():.4f}")
            print(f"  Context Brightness: {df['ctx_brightness'].mean():.4f} ± {df['ctx_brightness'].std():.4f}")
            print(f"  Context Edge Content: {df['ctx_edge_content'].mean():.4f} ± {df['ctx_edge_content'].std():.4f}")
            print(f"  Context Optical Flow: {df['ctx_optical_flow_magnitude'].mean():.4f} ± {df['ctx_optical_flow_magnitude'].std():.4f}")
            
            for pred_tokens in predict_tokens_list:
                predict_frames = pred_tokens * TUBELET_SIZE
                cos_col = f'pred_{pred_tokens}_cos_mean'
                l2_col = f'pred_{pred_tokens}_l2_mean'
                cos_max_col = f'pred_{pred_tokens}_cos_max'
                l2_max_col = f'pred_{pred_tokens}_l2_max'
                if cos_col in df.columns:
                    print(f"\n  Prediction {predict_tokens} tokens ({predict_frames} frames, {predict_frames/24:.2f}s):")
                    print(f"    L2 Distance (mean): {df[l2_col].mean():.4f} ± {df[l2_col].std():.4f}")
                    if l2_max_col in df.columns:
                        print(f"    L2 Distance (max):  {df[l2_max_col].mean():.4f} ± {df[l2_max_col].std():.4f}")
                    print(f"    Cosine Distance (mean): {df[cos_col].mean():.4f} ± {df[cos_col].std():.4f}")
                    if cos_max_col in df.columns:
                        print(f"    Cosine Distance (max):  {df[cos_max_col].mean():.4f} ± {df[cos_max_col].std():.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error processing {video_stem}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(args):
    """Main execution function."""
    print(f"\nStarting V-JEPA Sliding Window Analysis (v3 - Token-Aware)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Frame order: {args.order}")
    
    # Check token budget
    valid_tokens, warnings = check_token_budget(
        CONTEXT_TOKENS, 
        PREDICT_TOKENS_LIST,
        auto_clamp=not args.no_auto_clamp
    )
    
    for warning in warnings:
        print(warning)
    
    if valid_tokens != PREDICT_TOKENS_LIST and not args.no_auto_clamp:
        print(f"\nUsing clamped prediction tokens: {valid_tokens}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nLoading model: {HF_MODEL_NAME}")
    proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
    print("Model loaded successfully")
    
    # Create output directory
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process video(s)
    processed_count = 0
    failed_count = 0
    
    # Handle multiple videos (comma-separated)
    video_names = [v.strip() for v in args.video.split(',')]
    
    for video_name in video_names:
        video_path = VIDEO_PATH_TEMPLATE.format(video_name=video_name)
        
        if os.path.exists(video_path):
            success = run_sliding_analysis(
                mdl, proc, video_path, 
                frame_order=args.order,
                predict_tokens_list=valid_tokens if not args.no_auto_clamp else PREDICT_TOKENS_LIST
            )
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
    parser = argparse.ArgumentParser(
        description="V-JEPA2 Sliding Window Predictability Analysis (Token-Aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vjepa2_sliding_window_v3.py --video foetsie --order forward
  python vjepa2_sliding_window_v3.py --video foetsie --order backward --no-auto-clamp
  python vjepa2_sliding_window_v3.py --video "foetsie,trap" --order shuffled
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        default='foetsie',
        help='Video name(s) to process (comma-separated for multiple). Default: foetsie'
    )
    
    parser.add_argument(
        '--order',
        type=str,
        choices=['forward', 'backward', 'shuffled'],
        default='forward',
        help='Frame ordering: forward (normal), backward (reversed), or shuffled (random). Default: forward'
    )
    
    parser.add_argument(
        '--no-auto-clamp',
        action='store_true',
        help='Disable automatic token budget clamping (will use requested tokens even if they exceed budget)'
    )
    
    args = parser.parse_args()
    main(args)













# """
# V-JEPA2 Sliding Window Predictability Analysis
# ===============================================

# WHAT THIS SCRIPT DOES:
# 1. Slides a moving window through the entire video
# 2. For each window position, extracts:
#    - Context frames: the past N frames (used for prediction)
#    - Target frames: future M frames (what we're trying to predict)
# 3. Encodes context and target frames independently
# 4. Predicts future latents from context only
# 5. Measures prediction error at different future horizons
# 6. Correlates prediction error with visual statistics

# KEY INSIGHT:
# By sliding the window, we get frame-by-frame estimates of "predictability"
# - Frames with high motion/change → higher prediction error
# - Frames with stable content → lower prediction error

# This allows analysis like: "Are fast-moving scenes harder to predict?"

# USAGE:
#   python vjepa2_sliding_window_v2.py --video foetsie --order forward
#   python vjepa2_sliding_window_v2.py --video foetsie --order backward
#   python vjepa2_sliding_window_v2.py --video foetsie --order shuffled
# """

# import os
# import cv2
# import torch
# import numpy as np
# import pandas as pd
# import argparse
# from pathlib import Path
# from datetime import datetime
# from PIL import Image
# import torch.nn.functional as F
# from transformers import AutoModel, AutoVideoProcessor
# import matplotlib
# matplotlib.use('Agg')

# # --- Configuration ---
# HF_MODEL_NAME     = "facebook/vjepa2-vitg-fpc64-384"
# CONTEXT_TOKENS    = 10   
# PREDICT_TOKENS_LIST = [4, 8, 12]  # Different future prediction horizons
# TUBELET_SIZE      = 2    
# STRIDE            = 2  
# OUTPUT_BASE_DIR   = Path("vjepa_results_sliding")
# VIDEO_PATH_TEMPLATE = "/project/3018078.02/MEG_ingmar/shorts/{video_name}.mp4"

# # --- Visual Statistics Functions ---

# def compute_rms_contrast(frame):
#     """Compute RMS (root mean square) contrast of a frame."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#     return np.sqrt(np.mean(gray ** 2))

# def compute_brightness(frame):
#     """Compute mean brightness of a frame."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#     return np.mean(gray)

# def compute_edge_content(frame):
#     """Compute edge density using Canny edge detection."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     return np.mean(edges) / 255.0

# def compute_optical_flow_magnitude(frame1, frame2):
#     """Compute mean optical flow magnitude between two consecutive frames."""
#     gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
#     gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
#     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
#     return np.mean(magnitude)

# def compute_temporal_variance(frames_list):
#     """Compute temporal variance across a sequence of frames."""
#     if len(frames_list) < 2:
#         return 0.0
    
#     grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for f in frames_list]
#     temporal_var = np.var([np.mean(g) for g in grays])
#     return temporal_var

# def compute_visual_statistics(frame, prev_frame=None):
#     """Compute all visual statistics for a frame."""
#     stats = {
#         'rms_contrast': compute_rms_contrast(frame),
#         'brightness': compute_brightness(frame),
#         'edge_content': compute_edge_content(frame),
#     }
    
#     if prev_frame is not None:
#         stats['optical_flow_magnitude'] = compute_optical_flow_magnitude(prev_frame, frame)
#     else:
#         stats['optical_flow_magnitude'] = np.nan
    
#     return stats

# # --- Helper Functions ---

# def apply_frame_order(frames, order):
#     """
#     Apply frame ordering transformation.
    
#     Args:
#         frames: List of frame arrays
#         order: 'forward', 'backward', or 'shuffled'
    
#     Returns:
#         Transformed frames list
#     """
#     if order == 'forward':
#         return frames
#     elif order == 'backward':
#         return frames[::-1]
#     elif order == 'shuffled':
#         shuffled = frames.copy()
#         np.random.shuffle(shuffled)
#         return shuffled
#     else:
#         raise ValueError(f"Unknown frame order: {order}")

# def calculate_metrics(predicted, ground_truth):
#     """
#     Calculate cosine and L2 distances between predicted and ground truth latents.
    
#     Args:
#         predicted: [T, P, D] predicted latents
#         ground_truth: [T, P, D] ground truth latents
    
#     Returns:
#         cos_dist: cosine distance per timestep
#         l2_dist: L2 distance per timestep
#     """
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
#     """
#     Extract V-JEPA latent features from a sequence of frames.
    
#     Args:
#         model: V-JEPA2 model
#         processor: V-JEPA2 video processor
#         frames: List of numpy arrays [H, W, 3] in RGB
    
#     Returns:
#         feats: [T_out, P, D] latent features
#     """
#     video = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames])
#     inputs = processor(video, return_tensors="pt")
#     pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
#     feats = model.get_vision_features(pixel_values).cpu()
#     P = feats.shape[1] // (len(frames) // TUBELET_SIZE)
#     return feats.reshape(-1, P, feats.shape[-1])

# @torch.inference_mode()
# def predict_future_latents(model, context_latents, num_future_steps):
#     """
#     Predict future latents given context latents.
    
#     CRITICAL: Only context latents are visible to the predictor (causal prediction).
    
#     Args:
#         model: V-JEPA2 model with predictor
#         context_latents: [T_ctx, P, D] context latents only
#         num_future_steps: Number of future steps to predict
    
#     Returns:
#         predicted_latents: [num_future_steps, P, D] predicted future latents
#     """
#     T_ctx, P, D = context_latents.shape
#     total_len = T_ctx + num_future_steps
#     predictor = model.predictor
#     device = next(model.parameters()).device
    
#     # ============================================
#     # FIXED: Proper masking for causal prediction
#     # ============================================
#     # Context mask: 1 for positions with real context, 0 for future (hidden)
#     context_mask = torch.ones(1, T_ctx * P, dtype=torch.int64, device=device)
#     context_mask = torch.cat([
#         context_mask,
#         torch.zeros(1, num_future_steps * P, dtype=torch.int64, device=device)
#     ], dim=1)
    
#     # Target mask: 1 for positions to predict
#     target_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
#     target_mask[:, T_ctx * P : (T_ctx + num_future_steps) * P] = 1
    
#     # Build padded sequence: [context latents | zeros for future]
#     z_padded = torch.zeros(1, total_len * P, D, device=device)
#     z_ctx = context_latents.reshape(1, -1, D).to(device)
#     z_padded[:, :T_ctx * P, :] = z_ctx
    
#     # Run predictor
#     out = predictor(
#         encoder_hidden_states=z_padded,
#         context_mask=[context_mask],  # Only context is visible
#         target_mask=[target_mask]      # Predict these positions
#     )
    
#     return out.last_hidden_state[:, T_ctx * P :, :].reshape(num_future_steps, P, D).cpu()

# def run_sliding_analysis(model, processor, video_path, frame_order='forward'):
#     """
#     Run sliding window analysis on video, computing predictability and visual stats.
    
#     For each window position:
#     1. Extract context frames and encode them independently
#     2. Extract target frames and encode them independently
#     3. Predict from context only
#     4. Measure error and correlate with visual statistics
    
#     Args:
#         model: V-JEPA2 model
#         processor: V-JEPA2 processor
#         video_path: Path to video file
#         frame_order: 'forward', 'backward', or 'shuffled'
#     """
#     video_stem = Path(video_path).stem
#     video_dir = OUTPUT_BASE_DIR / video_stem
    
#     try:
#         print(f"\n{'='*60}")
#         print(f"Processing: {video_stem} (order: {frame_order})")
#         print(f"{'='*60}")
        
#         # Load video
#         cap = cv2.VideoCapture(video_path)
#         total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Check if video is long enough
#         ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
#         min_frames_needed = ctx_f + max(PREDICT_TOKENS_LIST) * TUBELET_SIZE * STRIDE
        
#         if total_v_frames < min_frames_needed:
#             print(f"Skipping {video_stem}: Video too short ({total_v_frames} < {min_frames_needed} frames)")
#             cap.release()
#             return False
        
#         # Create output directory
#         video_dir.mkdir(parents=True, exist_ok=True)
        
#         # Load all frames
#         raw_frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             raw_frames.append(cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB))
#         cap.release()
        
#         print(f"Loaded {len(raw_frames)} frames")
        
#         # Apply frame order transformation
#         raw_frames = apply_frame_order(raw_frames, frame_order)
#         print(f"Applied frame order: {frame_order}")
        
#         # Prepare output data structure
#         frame_results = []
#         max_predict = max(PREDICT_TOKENS_LIST)
        
#         # Sliding window analysis
#         num_windows = len(raw_frames) - ctx_f - (max_predict * TUBELET_SIZE * STRIDE)
        
#         print(f"Running sliding window analysis ({num_windows} windows)...")
        
#         for window_idx in range(num_windows):
#             if (window_idx + 1) % max(1, num_windows // 10) == 0:
#                 print(f"  Progress: {window_idx + 1}/{num_windows} windows processed")
            
#             # Get context frames
#             ctx_start = window_idx
#             ctx_end = ctx_start + ctx_f
#             context_frames_raw = raw_frames[ctx_start:ctx_end:STRIDE]
            
#             # Get ground truth frames (future frames to predict)
#             gt_start = ctx_end
#             max_gt_end = gt_start + (max_predict * TUBELET_SIZE * STRIDE)
#             all_future_frames_raw = raw_frames[gt_start:max_gt_end:STRIDE]
            
#             # ============================================
#             # ============================================
#             # Encode context frames ALONE
#             # Ground truth will be encoded per-horizon below
#             # ============================================
#             try:
#                 c_lat = get_vjepa_latents(model, processor, context_frames_raw)
#             except Exception as e:
#                 print(f"    Warning: Context latent extraction failed at window {window_idx}: {e}")
#                 continue
            
#             # Compute visual statistics for context frames
#             vis_stats = {
#                 'frame_index': window_idx,
#             }
            
#             # Context visual stats
#             for stat_name in ['rms_contrast', 'brightness', 'edge_content', 'optical_flow_magnitude']:
#                 vis_stats[f'ctx_{stat_name}'] = []
            
#             prev_frame = None
#             for i, frame in enumerate(context_frames_raw):
#                 stats = compute_visual_statistics(frame, prev_frame)
#                 for stat_name, stat_val in stats.items():
#                     vis_stats[f'ctx_{stat_name}'].append(stat_val)
#                 prev_frame = frame
            
#             # Take mean of context stats
#             for stat_name in ['rms_contrast', 'brightness', 'edge_content', 'optical_flow_magnitude']:
#                 key = f'ctx_{stat_name}'
#                 if vis_stats[key]:
#                     vis_stats[key] = np.nanmean(vis_stats[key])
#                 else:
#                     vis_stats[key] = np.nan
            
#             # Compute predictions for different future horizons
#             for predict_steps in PREDICT_TOKENS_LIST:
#                 # ============================================
#                 # CRITICAL FIX: Encode ground truth for THIS horizon only
#                 # Don't reuse all_future_lat[:predict_steps] because:
#                 # - Positional encodings differ when part of 4-frame vs 12-frame sequence
#                 # - Context dependencies bias the latents
#                 # ============================================
#                 try:
#                     gt_frames_for_horizon = all_future_frames_raw[:predict_steps]
#                     g_lat = get_vjepa_latents(model, processor, gt_frames_for_horizon)
#                 except Exception as e:
#                     print(f"    Warning: Ground truth encoding failed for {predict_steps}-step horizon at window {window_idx}: {e}")
#                     vis_stats[f'pred_{predict_steps}_cos_mean'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_l2_mean'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_cos_std'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_l2_std'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_cos_max'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_l2_max'] = np.nan
#                     for t in range(predict_steps):
#                         vis_stats[f'pred_{predict_steps}_l2_frame_{t}'] = np.nan
#                     continue
                
#                 # Predict future latents from context only
#                 p_lat = predict_future_latents(model, c_lat, predict_steps)
                
#                 # Calculate metrics
#                 cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
                
#                 # Store metrics: mean, std, and max (as suggested in V-JEPA papers)
#                 vis_stats[f'pred_{predict_steps}_cos_mean'] = np.mean(cos_dist)
#                 vis_stats[f'pred_{predict_steps}_l2_mean'] = np.mean(l2_dist)
#                 vis_stats[f'pred_{predict_steps}_cos_std'] = np.std(cos_dist)
#                 vis_stats[f'pred_{predict_steps}_l2_std'] = np.std(l2_dist)
                
#                 # Maximum error (useful for violation of expectation style analysis)
#                 vis_stats[f'pred_{predict_steps}_cos_max'] = np.max(cos_dist)
#                 vis_stats[f'pred_{predict_steps}_l2_max'] = np.max(l2_dist)
                
#                 # Per-frame error (to see if error monotonically degrades)
#                 for t in range(len(l2_dist)):
#                     vis_stats[f'pred_{predict_steps}_l2_frame_{t}'] = l2_dist[t]
            
#             frame_results.append(vis_stats)
        
#         # Save results to CSV
#         if frame_results:
#             df = pd.DataFrame(frame_results)
#             csv_path = video_dir / f"frame_metrics_{frame_order}.csv"
#             df.to_csv(csv_path, index=False)
#             print(f"\nResults saved to: {csv_path}")
            
#             # Print summary statistics
#             print(f"\nSummary Statistics for {video_stem} ({frame_order}):")
#             print(f"  Frames analyzed: {len(frame_results)}")
#             print(f"  Context RMS Contrast: {df['ctx_rms_contrast'].mean():.4f} ± {df['ctx_rms_contrast'].std():.4f}")
#             print(f"  Context Brightness: {df['ctx_brightness'].mean():.4f} ± {df['ctx_brightness'].std():.4f}")
#             print(f"  Context Edge Content: {df['ctx_edge_content'].mean():.4f} ± {df['ctx_edge_content'].std():.4f}")
#             print(f"  Context Optical Flow: {df['ctx_optical_flow_magnitude'].mean():.4f} ± {df['ctx_optical_flow_magnitude'].std():.4f}")
            
#             for pred_steps in PREDICT_TOKENS_LIST:
#                 cos_col = f'pred_{pred_steps}_cos_mean'
#                 l2_col = f'pred_{pred_steps}_l2_mean'
#                 cos_max_col = f'pred_{pred_steps}_cos_max'
#                 l2_max_col = f'pred_{pred_steps}_l2_max'
#                 if cos_col in df.columns:
#                     print(f"\n  Prediction {pred_steps} steps:")
#                     print(f"    L2 Distance (mean): {df[l2_col].mean():.4f} ± {df[l2_col].std():.4f}")
#                     if l2_max_col in df.columns:
#                         print(f"    L2 Distance (max):  {df[l2_max_col].mean():.4f} ± {df[l2_max_col].std():.4f}")
#                     print(f"    Cosine Distance (mean): {df[cos_col].mean():.4f} ± {df[cos_col].std():.4f}")
#                     if cos_max_col in df.columns:
#                         print(f"    Cosine Distance (max):  {df[cos_max_col].mean():.4f} ± {df[cos_max_col].std():.4f}")
                    
#                     # Show per-frame errors if available
#                     per_frame_cols = [c for c in df.columns if c.startswith(f'pred_{pred_steps}_l2_frame_')]
#                     if per_frame_cols:
#                         print(f"    Per-frame L2 distances:")
#                         for frame_idx in range(pred_steps):
#                             frame_col = f'pred_{pred_steps}_l2_frame_{frame_idx}'
#                             if frame_col in df.columns:
#                                 print(f"      Frame {frame_idx}: {df[frame_col].mean():.4f} ± {df[frame_col].std():.4f}")
        
#         return True
    
#     except Exception as e:
#         print(f"Error processing {video_stem}: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def main(args):
#     """Main execution function."""
#     print(f"\nStarting V-JEPA Sliding Window Analysis")
#     print(f"Timestamp: {datetime.now().isoformat()}")
#     print(f"Output directory: {OUTPUT_BASE_DIR}")
#     print(f"Frame order: {args.order}")
    
#     # Setup device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # Load model
#     print(f"\nLoading model: {HF_MODEL_NAME}")
#     proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
#     mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
#     print("Model loaded successfully")
    
#     # Create output directory
#     OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
#     # Process video(s)
#     processed_count = 0
#     failed_count = 0
    
#     # Handle multiple videos (comma-separated)
#     video_names = [v.strip() for v in args.video.split(',')]
    
#     for video_name in video_names:
#         video_path = VIDEO_PATH_TEMPLATE.format(video_name=video_name)
        
#         if os.path.exists(video_path):
#             success = run_sliding_analysis(mdl, proc, video_path, frame_order=args.order)
#             if success:
#                 processed_count += 1
#             else:
#                 failed_count += 1
#         else:
#             print(f"Video not found: {video_path}")
#             failed_count += 1
    
#     # Final summary
#     print(f"\n{'='*60}")
#     print(f"Analysis Complete!")
#     print(f"Successfully processed: {processed_count} videos")
#     print(f"Failed: {failed_count} videos")
#     print(f"Timestamp: {datetime.now().isoformat()}")
#     print(f"Results in: {OUTPUT_BASE_DIR}")
#     print(f"{'='*60}\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="V-JEPA2 Sliding Window Predictability Analysis",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python vjepa2_sliding_window_v2.py --video foetsie --order forward
#   python vjepa2_sliding_window_v2.py --video foetsie --order backward
#   python vjepa2_sliding_window_v2.py --video foetsie --order shuffled
#   python vjepa2_sliding_window_v2.py --video "foetsie,trap,titanic_bound" --order forward
#         """
#     )
    
#     parser.add_argument(
#         '--video',
#         type=str,
#         default='foetsie',
#         help='Video name(s) to process (comma-separated for multiple). Default: foetsie'
#     )
    
#     parser.add_argument(
#         '--order',
#         type=str,
#         choices=['forward', 'backward', 'shuffled'],
#         default='forward',
#         help='Frame ordering: forward (normal), backward (reversed), or shuffled (random). Default: forward'
#     )
    
#     args = parser.parse_args()
#     main(args)
















# """
# V-JEPA2 Sliding Window Predictability Analysis
# ===============================================

# WHAT THIS SCRIPT DOES:
# 1. Slides a moving window through the entire video
# 2. For each window position, extracts:
#    - Context frames: the past N frames (used for prediction)
#    - Target frames: future M frames (what we're trying to predict)
# 3. Encodes context and target frames independently
# 4. Predicts future latents from context only
# 5. Measures prediction error at different future horizons
# 6. Correlates prediction error with visual statistics

# KEY INSIGHT:
# By sliding the window, we get frame-by-frame estimates of "predictability"
# - Frames with high motion/change → higher prediction error
# - Frames with stable content → lower prediction error

# This allows analysis like: "Are fast-moving scenes harder to predict?"

# USAGE:
#   python vjepa2_sliding_window_v2.py --video foetsie --order forward
#   python vjepa2_sliding_window_v2.py --video foetsie --order backward
#   python vjepa2_sliding_window_v2.py --video foetsie --order shuffled
# """

# import os
# import cv2
# import torch
# import numpy as np
# import pandas as pd
# import argparse
# from pathlib import Path
# from datetime import datetime
# from PIL import Image
# import torch.nn.functional as F
# from transformers import AutoModel, AutoVideoProcessor
# import matplotlib
# matplotlib.use('Agg')

# # --- Configuration ---
# HF_MODEL_NAME     = "facebook/vjepa2-vitg-fpc64-384"
# CONTEXT_TOKENS    = 10   
# PREDICT_TOKENS_LIST = [4, 8, 12]  # Different future prediction horizons
# TUBELET_SIZE      = 2    
# STRIDE            = 2  
# OUTPUT_BASE_DIR   = Path("vjepa_results_sliding")
# VIDEO_PATH_TEMPLATE = "/project/3018078.02/MEG_ingmar/shorts/{video_name}.mp4"

# # --- Visual Statistics Functions ---

# def compute_rms_contrast(frame):
#     """Compute RMS (root mean square) contrast of a frame."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#     return np.sqrt(np.mean(gray ** 2))

# def compute_brightness(frame):
#     """Compute mean brightness of a frame."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#     return np.mean(gray)

# def compute_edge_content(frame):
#     """Compute edge density using Canny edge detection."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     return np.mean(edges) / 255.0

# def compute_optical_flow_magnitude(frame1, frame2):
#     """Compute mean optical flow magnitude between two consecutive frames."""
#     gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
#     gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
#     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
#     return np.mean(magnitude)

# def compute_temporal_variance(frames_list):
#     """Compute temporal variance across a sequence of frames."""
#     if len(frames_list) < 2:
#         return 0.0
    
#     grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for f in frames_list]
#     temporal_var = np.var([np.mean(g) for g in grays])
#     return temporal_var

# def compute_visual_statistics(frame, prev_frame=None):
#     """Compute all visual statistics for a frame."""
#     stats = {
#         'rms_contrast': compute_rms_contrast(frame),
#         'brightness': compute_brightness(frame),
#         'edge_content': compute_edge_content(frame),
#     }
    
#     if prev_frame is not None:
#         stats['optical_flow_magnitude'] = compute_optical_flow_magnitude(prev_frame, frame)
#     else:
#         stats['optical_flow_magnitude'] = np.nan
    
#     return stats

# # --- Helper Functions ---

# def apply_frame_order(frames, order):
#     """
#     Apply frame ordering transformation.
    
#     Args:
#         frames: List of frame arrays
#         order: 'forward', 'backward', or 'shuffled'
    
#     Returns:
#         Transformed frames list
#     """
#     if order == 'forward':
#         return frames
#     elif order == 'backward':
#         return frames[::-1]
#     elif order == 'shuffled':
#         shuffled = frames.copy()
#         np.random.shuffle(shuffled)
#         return shuffled
#     else:
#         raise ValueError(f"Unknown frame order: {order}")

# def calculate_metrics(predicted, ground_truth):
#     """
#     Calculate cosine and L2 distances between predicted and ground truth latents.
    
#     Args:
#         predicted: [T, P, D] predicted latents
#         ground_truth: [T, P, D] ground truth latents
    
#     Returns:
#         cos_dist: cosine distance per timestep
#         l2_dist: L2 distance per timestep
#     """
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
#     """
#     Extract V-JEPA latent features from a sequence of frames.
    
#     Args:
#         model: V-JEPA2 model
#         processor: V-JEPA2 video processor
#         frames: List of numpy arrays [H, W, 3] in RGB
    
#     Returns:
#         feats: [T_out, P, D] latent features
#     """
#     video = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames])
#     inputs = processor(video, return_tensors="pt")
#     pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
#     feats = model.get_vision_features(pixel_values).cpu()
#     P = feats.shape[1] // (len(frames) // TUBELET_SIZE)
#     return feats.reshape(-1, P, feats.shape[-1])

# @torch.inference_mode()
# def predict_future_latents(model, context_latents, num_future_steps):
#     """
#     Predict future latents given context latents.
    
#     CRITICAL: Only context latents are visible to the predictor (causal prediction).
    
#     Args:
#         model: V-JEPA2 model with predictor
#         context_latents: [T_ctx, P, D] context latents only
#         num_future_steps: Number of future steps to predict
    
#     Returns:
#         predicted_latents: [num_future_steps, P, D] predicted future latents
#     """
#     T_ctx, P, D = context_latents.shape
#     total_len = T_ctx + num_future_steps
#     predictor = model.predictor
#     device = next(model.parameters()).device
    
#     # ============================================
#     # FIXED: Proper masking for causal prediction
#     # ============================================
#     # Context mask: 1 for positions with real context, 0 for future (hidden)
#     context_mask = torch.ones(1, T_ctx * P, dtype=torch.int64, device=device)
#     context_mask = torch.cat([
#         context_mask,
#         torch.zeros(1, num_future_steps * P, dtype=torch.int64, device=device)
#     ], dim=1)
    
#     # Target mask: 1 for positions to predict
#     target_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
#     target_mask[:, T_ctx * P : (T_ctx + num_future_steps) * P] = 1
    
#     # Build padded sequence: [context latents | zeros for future]
#     z_padded = torch.zeros(1, total_len * P, D, device=device)
#     z_ctx = context_latents.reshape(1, -1, D).to(device)
#     z_padded[:, :T_ctx * P, :] = z_ctx
    
#     # Run predictor
#     out = predictor(
#         encoder_hidden_states=z_padded,
#         context_mask=[context_mask],  # Only context is visible
#         target_mask=[target_mask]      # Predict these positions
#     )
    
#     return out.last_hidden_state[:, T_ctx * P :, :].reshape(num_future_steps, P, D).cpu()

# def run_sliding_analysis(model, processor, video_path, frame_order='forward'):
#     """
#     Run sliding window analysis on video, computing predictability and visual stats.
    
#     For each window position:
#     1. Extract context frames and encode them independently
#     2. Extract target frames and encode them independently
#     3. Predict from context only
#     4. Measure error and correlate with visual statistics
    
#     Args:
#         model: V-JEPA2 model
#         processor: V-JEPA2 processor
#         video_path: Path to video file
#         frame_order: 'forward', 'backward', or 'shuffled'
#     """
#     video_stem = Path(video_path).stem
#     video_dir = OUTPUT_BASE_DIR / video_stem
    
#     try:
#         print(f"\n{'='*60}")
#         print(f"Processing: {video_stem} (order: {frame_order})")
#         print(f"{'='*60}")
        
#         # Load video
#         cap = cv2.VideoCapture(video_path)
#         total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Check if video is long enough
#         ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
#         min_frames_needed = ctx_f + max(PREDICT_TOKENS_LIST) * TUBELET_SIZE * STRIDE
        
#         if total_v_frames < min_frames_needed:
#             print(f"Skipping {video_stem}: Video too short ({total_v_frames} < {min_frames_needed} frames)")
#             cap.release()
#             return False
        
#         # Create output directory
#         video_dir.mkdir(parents=True, exist_ok=True)
        
#         # Load all frames
#         raw_frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             raw_frames.append(cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB))
#         cap.release()
        
#         print(f"Loaded {len(raw_frames)} frames")
        
#         # Apply frame order transformation
#         raw_frames = apply_frame_order(raw_frames, frame_order)
#         print(f"Applied frame order: {frame_order}")
        
#         # Prepare output data structure
#         frame_results = []
#         max_predict = max(PREDICT_TOKENS_LIST)
        
#         # Sliding window analysis
#         num_windows = len(raw_frames) - ctx_f - (max_predict * TUBELET_SIZE * STRIDE)
        
#         print(f"Running sliding window analysis ({num_windows} windows)...")
        
#         for window_idx in range(num_windows):
#             if (window_idx + 1) % max(1, num_windows // 10) == 0:
#                 print(f"  Progress: {window_idx + 1}/{num_windows} windows processed")
            
#             # Get context frames
#             ctx_start = window_idx
#             ctx_end = ctx_start + ctx_f
#             context_frames_raw = raw_frames[ctx_start:ctx_end:STRIDE]
            
#             # Get ground truth frames (future frames to predict)
#             gt_start = ctx_end
#             max_gt_end = gt_start + (max_predict * TUBELET_SIZE * STRIDE)
#             all_future_frames_raw = raw_frames[gt_start:max_gt_end:STRIDE]
            
#             # ============================================
#             # KEY FIX #1: Encode context and target separately
#             # ============================================
#             try:
#                 # Encode context frames ALONE
#                 c_lat = get_vjepa_latents(model, processor, context_frames_raw)
                
#                 # Encode ALL future frames ALONE
#                 # (We'll slice out only what we need for each prediction horizon)
#                 all_future_lat = get_vjepa_latents(model, processor, all_future_frames_raw)
#             except Exception as e:
#                 print(f"    Warning: Latent extraction failed at window {window_idx}: {e}")
#                 continue
            
#             # Compute visual statistics for context frames
#             vis_stats = {
#                 'frame_index': window_idx,
#             }
            
#             # Context visual stats
#             for stat_name in ['rms_contrast', 'brightness', 'edge_content', 'optical_flow_magnitude']:
#                 vis_stats[f'ctx_{stat_name}'] = []
            
#             prev_frame = None
#             for i, frame in enumerate(context_frames_raw):
#                 stats = compute_visual_statistics(frame, prev_frame)
#                 for stat_name, stat_val in stats.items():
#                     vis_stats[f'ctx_{stat_name}'].append(stat_val)
#                 prev_frame = frame
            
#             # Take mean of context stats
#             for stat_name in ['rms_contrast', 'brightness', 'edge_content', 'optical_flow_magnitude']:
#                 key = f'ctx_{stat_name}'
#                 if vis_stats[key]:
#                     vis_stats[key] = np.nanmean(vis_stats[key])
#                 else:
#                     vis_stats[key] = np.nan
            
#             # Compute predictions for different future horizons
#             for predict_steps in PREDICT_TOKENS_LIST:
#                 # Get ground truth latents for this prediction horizon
#                 # (encoded independently, not from full sequence)
#                 if predict_steps <= len(all_future_lat):
#                     g_lat = all_future_lat[:predict_steps]
#                 else:
#                     # Skip this horizon if we don't have enough frames
#                     # (avoid padding zeros into latents)
#                     vis_stats[f'pred_{predict_steps}_cos_mean'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_l2_mean'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_cos_std'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_l2_std'] = np.nan
#                     continue
                
#                 # Predict future latents from context only
#                 p_lat = predict_future_latents(model, c_lat, predict_steps)
                
#                 # Calculate metrics
#                 cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
                
#                 # Store mean and std metrics
#                 vis_stats[f'pred_{predict_steps}_cos_mean'] = np.mean(cos_dist)
#                 vis_stats[f'pred_{predict_steps}_l2_mean'] = np.mean(l2_dist)
#                 vis_stats[f'pred_{predict_steps}_cos_std'] = np.std(cos_dist)
#                 vis_stats[f'pred_{predict_steps}_l2_std'] = np.std(l2_dist)
            
#             frame_results.append(vis_stats)
        
#         # Save results to CSV
#         if frame_results:
#             df = pd.DataFrame(frame_results)
#             csv_path = video_dir / f"frame_metrics_{frame_order}.csv"
#             df.to_csv(csv_path, index=False)
#             print(f"\nResults saved to: {csv_path}")
            
#             # Print summary statistics
#             print(f"\nSummary Statistics for {video_stem} ({frame_order}):")
#             print(f"  Frames analyzed: {len(frame_results)}")
#             print(f"  Context RMS Contrast: {df['ctx_rms_contrast'].mean():.4f} ± {df['ctx_rms_contrast'].std():.4f}")
#             print(f"  Context Brightness: {df['ctx_brightness'].mean():.4f} ± {df['ctx_brightness'].std():.4f}")
#             print(f"  Context Edge Content: {df['ctx_edge_content'].mean():.4f} ± {df['ctx_edge_content'].std():.4f}")
#             print(f"  Context Optical Flow: {df['ctx_optical_flow_magnitude'].mean():.4f} ± {df['ctx_optical_flow_magnitude'].std():.4f}")
            
#             for pred_steps in PREDICT_TOKENS_LIST:
#                 cos_col = f'pred_{pred_steps}_cos_mean'
#                 l2_col = f'pred_{pred_steps}_l2_mean'
#                 if cos_col in df.columns:
#                     print(f"\n  Prediction {pred_steps} steps:")
#                     print(f"    Cosine Distance: {df[cos_col].mean():.4f} ± {df[cos_col].std():.4f}")
#                     print(f"    L2 Distance: {df[l2_col].mean():.4f} ± {df[l2_col].std():.4f}")
        
#         return True
    
#     except Exception as e:
#         print(f"Error processing {video_stem}: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def main(args):
#     """Main execution function."""
#     print(f"\nStarting V-JEPA Sliding Window Analysis")
#     print(f"Timestamp: {datetime.now().isoformat()}")
#     print(f"Output directory: {OUTPUT_BASE_DIR}")
#     print(f"Frame order: {args.order}")
    
#     # Setup device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # Load model
#     print(f"\nLoading model: {HF_MODEL_NAME}")
#     proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
#     mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
#     print("Model loaded successfully")
    
#     # Create output directory
#     OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
#     # Process video(s)
#     processed_count = 0
#     failed_count = 0
    
#     # Handle multiple videos (comma-separated)
#     video_names = [v.strip() for v in args.video.split(',')]
    
#     for video_name in video_names:
#         video_path = VIDEO_PATH_TEMPLATE.format(video_name=video_name)
        
#         if os.path.exists(video_path):
#             success = run_sliding_analysis(mdl, proc, video_path, frame_order=args.order)
#             if success:
#                 processed_count += 1
#             else:
#                 failed_count += 1
#         else:
#             print(f"Video not found: {video_path}")
#             failed_count += 1
    
#     # Final summary
#     print(f"\n{'='*60}")
#     print(f"Analysis Complete!")
#     print(f"Successfully processed: {processed_count} videos")
#     print(f"Failed: {failed_count} videos")
#     print(f"Timestamp: {datetime.now().isoformat()}")
#     print(f"Results in: {OUTPUT_BASE_DIR}")
#     print(f"{'='*60}\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="V-JEPA2 Sliding Window Predictability Analysis",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python vjepa2_sliding_window_v2.py --video foetsie --order forward
#   python vjepa2_sliding_window_v2.py --video foetsie --order backward
#   python vjepa2_sliding_window_v2.py --video foetsie --order shuffled
#   python vjepa2_sliding_window_v2.py --video "foetsie,trap,titanic_bound" --order forward
#         """
#     )
    
#     parser.add_argument(
#         '--video',
#         type=str,
#         default='foetsie',
#         help='Video name(s) to process (comma-separated for multiple). Default: foetsie'
#     )
    
#     parser.add_argument(
#         '--order',
#         type=str,
#         choices=['forward', 'backward', 'shuffled'],
#         default='forward',
#         help='Frame ordering: forward (normal), backward (reversed), or shuffled (random). Default: forward'
#     )
    
#     args = parser.parse_args()
#     main(args)











# """
# V-JEPA2 Sliding Window Predictability Analysis
# ===============================================

# WHAT THIS SCRIPT DOES:
# 1. Slides a moving window through the entire video
# 2. For each window position, extracts:
#    - Context frames: the past N frames (used for prediction)
#    - Target frames: future M frames (what we're trying to predict)
# 3. Encodes context and target frames independently
# 4. Predicts future latents from context only
# 5. Measures prediction error at different future horizons
# 6. Correlates prediction error with visual statistics

# KEY INSIGHT:
# By sliding the window, we get frame-by-frame estimates of "predictability"
# - Frames with high motion/change → higher prediction error
# - Frames with stable content → lower prediction error

# This allows analysis like: "Are fast-moving scenes harder to predict?"
# """

# import os
# import cv2
# import torch
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from datetime import datetime
# from PIL import Image
# import torch.nn.functional as F
# from transformers import AutoModel, AutoVideoProcessor
# import matplotlib
# matplotlib.use('Agg')

# # --- Configuration ---
# HF_MODEL_NAME     = "facebook/vjepa2-vitg-fpc64-384"
# CONTEXT_TOKENS    = 10   
# PREDICT_TOKENS_LIST = [4, 8, 12]  # Different future prediction horizons
# TUBELET_SIZE      = 2    
# STRIDE            = 2  
# OUTPUT_BASE_DIR   = Path("vjepa_results_sliding")
# # VIDEO_NAMES       = ["deurinhuis", "trap", "titanic_bound", "iglo", "kanon", "sprongwagon", "geiser", 
#                     #  "klepdonder", "gevelstort", "foetsie", "uitzichtdoek", "yenga", "whackamole", 
#                     #  "trein_portret", "gooi_krat", "bw_testclip_pluim", "bw_testclip_lazeren"]
# # VIDEO_NAMES = ["ProjectAttention_movie_part21_24Hz"] # one longer one
# # VIDEO_NAMES = ["longclip_part42_bw"] # one longer one
# VIDEO_NAMES = ["foetsie"] # one longer one

# # --- Visual Statistics Functions ---

# def compute_rms_contrast(frame):
#     """Compute RMS (root mean square) contrast of a frame."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#     return np.sqrt(np.mean(gray ** 2))

# def compute_brightness(frame):
#     """Compute mean brightness of a frame."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#     return np.mean(gray)

# def compute_edge_content(frame):
#     """Compute edge density using Canny edge detection."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     return np.mean(edges) / 255.0

# def compute_optical_flow_magnitude(frame1, frame2):
#     """Compute mean optical flow magnitude between two consecutive frames."""
#     gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
#     gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
#     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
#     return np.mean(magnitude)

# def compute_temporal_variance(frames_list):
#     """Compute temporal variance across a sequence of frames."""
#     if len(frames_list) < 2:
#         return 0.0
    
#     grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for f in frames_list]
#     temporal_var = np.var([np.mean(g) for g in grays])
#     return temporal_var

# def compute_visual_statistics(frame, prev_frame=None):
#     """Compute all visual statistics for a frame."""
#     stats = {
#         'rms_contrast': compute_rms_contrast(frame),
#         'brightness': compute_brightness(frame),
#         'edge_content': compute_edge_content(frame),
#     }
    
#     if prev_frame is not None:
#         stats['optical_flow_magnitude'] = compute_optical_flow_magnitude(prev_frame, frame)
#     else:
#         stats['optical_flow_magnitude'] = np.nan
    
#     return stats

# # --- Helper Functions ---

# def calculate_metrics(predicted, ground_truth):
#     """
#     Calculate cosine and L2 distances between predicted and ground truth latents.
    
#     Args:
#         predicted: [T, P, D] predicted latents
#         ground_truth: [T, P, D] ground truth latents
    
#     Returns:
#         cos_dist: cosine distance per timestep
#         l2_dist: L2 distance per timestep
#     """
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
#     """
#     Extract V-JEPA latent features from a sequence of frames.
    
#     Args:
#         model: V-JEPA2 model
#         processor: V-JEPA2 video processor
#         frames: List of numpy arrays [H, W, 3] in RGB
    
#     Returns:
#         feats: [T_out, P, D] latent features
#     """
#     video = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames])
#     inputs = processor(video, return_tensors="pt")
#     pixel_values = inputs["pixel_values_videos"].to(next(model.parameters()).device)
#     feats = model.get_vision_features(pixel_values).cpu()
#     P = feats.shape[1] // (len(frames) // TUBELET_SIZE)
#     return feats.reshape(-1, P, feats.shape[-1])

# @torch.inference_mode()
# def predict_future_latents(model, context_latents, num_future_steps):
#     """
#     Predict future latents given context latents.
    
#     CRITICAL: Only context latents are visible to the predictor (causal prediction).
    
#     Args:
#         model: V-JEPA2 model with predictor
#         context_latents: [T_ctx, P, D] context latents only
#         num_future_steps: Number of future steps to predict
    
#     Returns:
#         predicted_latents: [num_future_steps, P, D] predicted future latents
#     """
#     T_ctx, P, D = context_latents.shape
#     total_len = T_ctx + num_future_steps
#     predictor = model.predictor
#     device = next(model.parameters()).device
    
#     # ============================================
#     # FIXED: Proper masking for causal prediction
#     # ============================================
#     # Context mask: 1 for positions with real context, 0 for future (hidden)
#     context_mask = torch.ones(1, T_ctx * P, dtype=torch.int64, device=device)
#     context_mask = torch.cat([
#         context_mask,
#         torch.zeros(1, num_future_steps * P, dtype=torch.int64, device=device)
#     ], dim=1)
    
#     # Target mask: 1 for positions to predict
#     target_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
#     target_mask[:, T_ctx * P : (T_ctx + num_future_steps) * P] = 1
    
#     # Build padded sequence: [context latents | zeros for future]
#     z_padded = torch.zeros(1, total_len * P, D, device=device)
#     z_ctx = context_latents.reshape(1, -1, D).to(device)
#     z_padded[:, :T_ctx * P, :] = z_ctx
    
#     # Run predictor
#     out = predictor(
#         encoder_hidden_states=z_padded,
#         context_mask=[context_mask],  # Only context is visible
#         target_mask=[target_mask]      # Predict these positions
#     )
    
#     return out.last_hidden_state[:, T_ctx * P :, :].reshape(num_future_steps, P, D).cpu()

# def run_sliding_analysis(model, processor, video_path):
#     """
#     Run sliding window analysis on video, computing predictability and visual stats.
    
#     For each window position:
#     1. Extract context frames and encode them independently
#     2. Extract target frames and encode them independently
#     3. Predict from context only
#     4. Measure error and correlate with visual statistics
#     """
#     video_stem = Path(video_path).stem
#     video_dir = OUTPUT_BASE_DIR / video_stem
    
#     try:
#         print(f"\n{'='*60}")
#         print(f"Processing: {video_stem}")
#         print(f"{'='*60}")
        
#         # Load video
#         cap = cv2.VideoCapture(video_path)
#         total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Check if video is long enough
#         ctx_f = CONTEXT_TOKENS * TUBELET_SIZE * STRIDE
#         min_frames_needed = ctx_f + max(PREDICT_TOKENS_LIST) * TUBELET_SIZE * STRIDE
        
#         if total_v_frames < min_frames_needed:
#             print(f"Skipping {video_stem}: Video too short ({total_v_frames} < {min_frames_needed} frames)")
#             cap.release()
#             return False
        
#         # Create output directory
#         video_dir.mkdir(parents=True, exist_ok=True)
        
#         # Load all frames
#         raw_frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             raw_frames.append(cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB))
#         cap.release()
        
#         print(f"Loaded {len(raw_frames)} frames")
        
#         # Prepare output data structure
#         frame_results = []
#         max_predict = max(PREDICT_TOKENS_LIST)
        
#         # Sliding window analysis
#         num_windows = len(raw_frames) - ctx_f - (max_predict * TUBELET_SIZE * STRIDE)
        
#         print(f"Running sliding window analysis ({num_windows} windows)...")
        
#         for window_idx in range(num_windows):
#             if (window_idx + 1) % max(1, num_windows // 10) == 0:
#                 print(f"  Progress: {window_idx + 1}/{num_windows} windows processed")
            
#             # Get context frames
#             ctx_start = window_idx
#             ctx_end = ctx_start + ctx_f
#             context_frames_raw = raw_frames[ctx_start:ctx_end:STRIDE]
            
#             # Get ground truth frames (future frames to predict)
#             gt_start = ctx_end
#             max_gt_end = gt_start + (max_predict * TUBELET_SIZE * STRIDE)
#             all_future_frames_raw = raw_frames[gt_start:max_gt_end:STRIDE]
            
#             # ============================================
#             # KEY FIX #1: Encode context and target separately
#             # ============================================
#             try:
#                 # Encode context frames ALONE
#                 c_lat = get_vjepa_latents(model, processor, context_frames_raw)
                
#                 # Encode ALL future frames ALONE
#                 # (We'll slice out only what we need for each prediction horizon)
#                 all_future_lat = get_vjepa_latents(model, processor, all_future_frames_raw)
#             except Exception as e:
#                 print(f"    Warning: Latent extraction failed at window {window_idx}: {e}")
#                 continue
            
#             # Compute visual statistics for context frames
#             vis_stats = {
#                 'frame_index': window_idx,
#             }
            
#             # Context visual stats
#             for stat_name in ['rms_contrast', 'brightness', 'edge_content', 'optical_flow_magnitude']:
#                 vis_stats[f'ctx_{stat_name}'] = []
            
#             prev_frame = None
#             for i, frame in enumerate(context_frames_raw):
#                 stats = compute_visual_statistics(frame, prev_frame)
#                 for stat_name, stat_val in stats.items():
#                     vis_stats[f'ctx_{stat_name}'].append(stat_val)
#                 prev_frame = frame
            
#             # Take mean of context stats
#             for stat_name in ['rms_contrast', 'brightness', 'edge_content', 'optical_flow_magnitude']:
#                 key = f'ctx_{stat_name}'
#                 if vis_stats[key]:
#                     vis_stats[key] = np.nanmean(vis_stats[key])
#                 else:
#                     vis_stats[key] = np.nan
            
#             # Compute predictions for different future horizons
#             for predict_steps in PREDICT_TOKENS_LIST:
#                 # Get ground truth latents for this prediction horizon
#                 # (encoded independently, not from full sequence)
#                 if predict_steps <= len(all_future_lat):
#                     g_lat = all_future_lat[:predict_steps]
#                 else:
#                     # Skip this horizon if we don't have enough frames
#                     # (avoid padding zeros into latents)
#                     vis_stats[f'pred_{predict_steps}_cos_mean'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_l2_mean'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_cos_std'] = np.nan
#                     vis_stats[f'pred_{predict_steps}_l2_std'] = np.nan
#                     continue
                
#                 # Predict future latents from context only
#                 p_lat = predict_future_latents(model, c_lat, predict_steps)
                
#                 # Calculate metrics
#                 cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
                
#                 # Store mean and std metrics
#                 vis_stats[f'pred_{predict_steps}_cos_mean'] = np.mean(cos_dist)
#                 vis_stats[f'pred_{predict_steps}_l2_mean'] = np.mean(l2_dist)
#                 vis_stats[f'pred_{predict_steps}_cos_std'] = np.std(cos_dist)
#                 vis_stats[f'pred_{predict_steps}_l2_std'] = np.std(l2_dist)
            
#             frame_results.append(vis_stats)
        
#         # Save results to CSV
#         if frame_results:
#             df = pd.DataFrame(frame_results)
#             csv_path = video_dir / "frame_metrics.csv"
#             df.to_csv(csv_path, index=False)
#             print(f"\nResults saved to: {csv_path}")
            
#             # Print summary statistics
#             print(f"\nSummary Statistics for {video_stem}:")
#             print(f"  Frames analyzed: {len(frame_results)}")
#             print(f"  Context RMS Contrast: {df['ctx_rms_contrast'].mean():.4f} ± {df['ctx_rms_contrast'].std():.4f}")
#             print(f"  Context Brightness: {df['ctx_brightness'].mean():.4f} ± {df['ctx_brightness'].std():.4f}")
#             print(f"  Context Edge Content: {df['ctx_edge_content'].mean():.4f} ± {df['ctx_edge_content'].std():.4f}")
#             print(f"  Context Optical Flow: {df['ctx_optical_flow_magnitude'].mean():.4f} ± {df['ctx_optical_flow_magnitude'].std():.4f}")
            
#             for pred_steps in PREDICT_TOKENS_LIST:
#                 cos_col = f'pred_{pred_steps}_cos_mean'
#                 l2_col = f'pred_{pred_steps}_l2_mean'
#                 if cos_col in df.columns:
#                     print(f"\n  Prediction {pred_steps} steps:")
#                     print(f"    Cosine Distance: {df[cos_col].mean():.4f} ± {df[cos_col].std():.4f}")
#                     print(f"    L2 Distance: {df[l2_col].mean():.4f} ± {df[l2_col].std():.4f}")
        
#         return True
    
#     except Exception as e:
#         print(f"Error processing {video_stem}: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def main():
#     """Main execution function."""
#     print(f"\nStarting V-JEPA Sliding Window Analysis")
#     print(f"Timestamp: {datetime.now().isoformat()}")
#     print(f"Output directory: {OUTPUT_BASE_DIR}")
    
#     # Setup device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # Load model
#     print(f"\nLoading model: {HF_MODEL_NAME}")
#     proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
#     mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
#     print("Model loaded successfully")
    
#     # Create output directory
#     OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
#     # Process videos
#     processed_count = 0
#     failed_count = 0
    
#     for video_name in VIDEO_NAMES:
#         video_path = f"/project/3018078.02/MEG_ingmar/shorts/{video_name}.mp4"
#         # video_path = f"/project/3018078.02/MEG_ingmar/{video_name}.mp4"
        
#         if os.path.exists(video_path):
#             success = run_sliding_analysis(mdl, proc, video_path)
#             if success:
#                 processed_count += 1
#             else:
#                 failed_count += 1
#         else:
#             print(f"Video not found: {video_path}")
#             failed_count += 1
    
#     # Final summary
#     print(f"\n{'='*60}")
#     print(f"Analysis Complete!")
#     print(f"Successfully processed: {processed_count} videos")
#     print(f"Failed: {failed_count} videos")
#     print(f"Timestamp: {datetime.now().isoformat()}")
#     print(f"Results in: {OUTPUT_BASE_DIR}")
#     print(f"{'='*60}\n")

# if __name__ == "__main__":
#     main()


