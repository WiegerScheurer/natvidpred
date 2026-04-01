import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- Configuration ---
HF_MODEL_NAME   = "facebook/vjepa2-vitg-fpc64-384"
CONTEXT_STEPS   = 10   # Number of latent steps for context
# HORIZONS        = [4, 8, 12] # Predict n steps into the future
HORIZONS        = [2, 4, 8] # Predict n steps into the future
TUBELET_SIZE    = 1   
STRIDE          = 1    
OUTPUT_DIR      = Path("vjepa_sliding_results")
VIDEO_DIR       = Path("/project/3018078.02/MEG_ingmar/shorts/")

# Visualization settings
FIGURE_DPI      = 150
FIGURE_SIZE     = (14, 6)

def get_low_level_stats(frame, prev_frame=None):
    """
    Computes change in local RMS contrast and Mean Optical Flow magnitude.
    
    Returns:
        rms_contrast: Mean change in local RMS across 32x32 patches
        optical_flow: Mean optical flow magnitude
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Compute local RMS contrast using sliding window
    window_size = 32
    stride = 16
    
    def local_rms(img):
        h, w = img.shape
        rms_map = []
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                patch = img[y:y+window_size, x:x+window_size]
                rms_map.append(np.std(patch))
        return np.array(rms_map)

    rms_contrast = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        rms_now = local_rms(gray)
        rms_prev = local_rms(prev_gray)
        # Align lengths
        min_len = min(len(rms_now), len(rms_prev))
        rms_change = np.abs(rms_now[:min_len] - rms_prev[:min_len])
        rms_contrast = np.mean(rms_change)

    flow_mag = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mag = np.mean(mag)

    return rms_contrast, flow_mag

@torch.inference_mode()
def extract_all_latents(model, processor, frames):
    """
    Extracts latents for the video tubelet-by-tubelet to avoid shape mismatches.
    
    Returns:
        feats: Tensor of shape [num_steps, num_patches, embedding_dim]
    """
    device = next(model.parameters()).device
    num_frames = len(frames)
    num_steps = num_frames // TUBELET_SIZE
    latents_list = []
    min_patches = None

    for i in range(num_steps):
        tubelet_frames = frames[i * TUBELET_SIZE : (i + 1) * TUBELET_SIZE]
        # Stack to [T, C, H, W] - convert to float to avoid uint8 issues
        video = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() 
            for f in tubelet_frames
        ])
        
        # Pass video directly to processor (without unsqueeze)
        # VJEPA2VideoProcessor handles batching internally
        inputs = processor(video, return_tensors="pt")
        pixel_values = inputs["pixel_values_videos"].to(device)
        
        # pixel_values is [1, 1, 1, T, C, H, W] - need to reshape
        # Reshape to [1, T, C, H, W] by removing extra batch dims
        if pixel_values.dim() == 7:
            pixel_values = pixel_values.squeeze(1).squeeze(1)  # Remove dims 1 and 2
        
        feats = model.get_vision_features(pixel_values).cpu()
        
        # Handle output shape: should be [1, P, D], squeeze batch dim
        if feats.dim() == 3:
            feats = feats.squeeze(0)
        
        latents_list.append(feats)
        
        if min_patches is None or feats.shape[0] < min_patches:
            min_patches = feats.shape[0]

    # Truncate all to min_patches to ensure stackability
    latents_list = [f[:min_patches, :] for f in latents_list]
    feats = torch.stack(latents_list)  # [num_steps, P, D]
    return feats

@torch.inference_mode()
def predict_at_horizon(model, context_latents, horizon):
    """
    Predicts latents a specific number of steps into the future.
    
    Args:
        context_latents: Shape [T_ctx, P, D] where T_ctx is context length
        horizon: Number of steps to predict into future
        
    Returns:
        Predicted latents of shape [horizon, P, D]
    """
    T_ctx, P, D = context_latents.shape
    total_len = T_ctx + horizon
    device = next(model.parameters()).device
    
    # Create masks for context and target regions
    ctx_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
    ctx_mask[:, :T_ctx * P] = 1
    tgt_mask = torch.zeros(1, total_len * P, dtype=torch.int64, device=device)
    tgt_mask[:, T_ctx * P:] = 1

    z_padded = torch.zeros(1, total_len * P, D, device=device)
    z_padded[:, :T_ctx * P, :] = context_latents.reshape(1, -1, D).to(device)

    out = model.predictor(encoder_hidden_states=z_padded, context_mask=[ctx_mask], target_mask=[tgt_mask])
    return out.last_hidden_state[:, T_ctx * P :, :].reshape(horizon, P, D).cpu()

def compute_prediction_error(pred, target):
    """
    Computes Euclidean distance prediction error (lower is better).
    
    Args:
        pred, target: Both shape [horizon, P, D]
        
    Returns:
        euclidean_error: Mean Euclidean distance across patches and horizon
    """
    eucl_dist = torch.norm(pred - target, dim=-1)  # [horizon, P]
    return eucl_dist.mean().item()

def compute_correlation_matrix(df):
    """
    Computes correlation matrix between all metrics.
    
    Returns:
        corr_matrix: Correlation dataframe
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    return corr_matrix

def compute_horizon_stats(df):
    """
    Computes mean and max values for each prediction horizon.
    
    Returns:
        stats_dict: Dictionary with 'mean' and 'max' keys, each containing
                   values for each horizon
    """
    stats_dict = {"mean": {}, "max": {}}
    for h in HORIZONS:
        col_name = f"unpredictability_h{h}_euclidean"
        if col_name in df.columns:
            stats_dict["mean"][h] = df[col_name].mean()
            stats_dict["max"][h] = df[col_name].max()
    return stats_dict

def plot_metrics_over_time(df, video_stem, frame_order, horizon_stats=None):
    """
    Creates a line plot of visual features and prediction errors over time.
    
    Args:
        df: DataFrame with metrics
        video_stem: Video name
        frame_order: Forward/backward/shuffled
        horizon_stats: Dictionary with mean/max stats per horizon
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.suptitle(f"Metrics Over Time: {video_stem} ({frame_order})", fontsize=14, fontweight='bold')
    
    # Plot 1: Low-level visual features
    ax = axes[0, 0]
    ax.plot(df['rms_contrast'], label='RMS Contrast Change', linewidth=1.5, alpha=0.7)
    ax.plot(df['optical_flow'], label='Optical Flow Magnitude', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Latent Step')
    ax.set_ylabel('Magnitude')
    ax.set_title('Low-Level Visual Features')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Euclidean distances across horizons
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(HORIZONS)))
    for idx, h in enumerate(HORIZONS):
        col_name = f"unpredictability_h{h}_euclidean"
        if col_name in df.columns:
            ax.plot(df[col_name], label=f'Horizon {h}', linewidth=1.5, alpha=0.7, color=colors[idx])
            # Add horizontal line for max value
            if horizon_stats and h in horizon_stats["max"]:
                max_val = horizon_stats["max"][h]
                ax.axhline(y=max_val, linestyle='--', alpha=0.5, color=colors[idx],
                          linewidth=1, label=f'Max H{h}: {max_val:.3f}')
    ax.set_xlabel('Latent Step')
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('V-JEPA2 Prediction Error (Euclidean)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of prediction errors
    ax = axes[1, 0]
    euclidean_cols = [f"unpredictability_h{h}_euclidean" for h in HORIZONS if f"unpredictability_h{h}_euclidean" in df.columns]
    box_data = [df[col].values for col in euclidean_cols]
    bp = ax.boxplot(box_data, labels=[f'H{h}' for h in HORIZONS], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('Distribution of Prediction Errors by Horizon')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Correlation heatmap (subset of key correlations)
    ax = axes[1, 1]
    corr_matrix = compute_correlation_matrix(df)
    # Select a subset of important correlations to visualize
    key_cols = ['rms_contrast', 'optical_flow'] + euclidean_cols
    key_cols = [col for col in key_cols if col in corr_matrix.columns]
    if len(key_cols) > 1:
        subset_corr = corr_matrix.loc[key_cols, key_cols]
        sns.heatmap(subset_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    ax=ax, cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
        ax.set_title('Metric Correlations')
    
    plt.tight_layout()
    return fig, horizon_stats

def plot_correlation_matrix(corr_matrix, video_stem, frame_order):
    """
    Creates a full correlation heatmap visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=ax, cbar_kws={'label': 'Pearson Correlation'}, vmin=-1, vmax=1,
                square=True)
    
    ax.set_title(f"Correlation Matrix: {video_stem} ({frame_order})", 
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def process_video(video_path, model, processor, frame_order):
    """
    Main processing function for a single video and frame order.
    """
    print(f"Processing: {video_path.name} ({frame_order})")

    # Load video frames
    cap = cv2.VideoCapture(str(video_path))
    raw_frames = []
    low_level_stats = []
    prev_f = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb)
        rms, flow = get_low_level_stats(frame_rgb, prev_f)
        low_level_stats.append({"rms_contrast": rms, "optical_flow": flow})
        prev_f = frame_rgb
    cap.release()

    # Apply frame order toggle
    import random
    if frame_order == "backward":
        raw_frames = raw_frames[::-1]
        low_level_stats = low_level_stats[::-1]
    elif frame_order == "shuffled":
        combined = list(zip(raw_frames, low_level_stats))
        random.shuffle(combined)
        raw_frames, low_level_stats = zip(*combined)
        raw_frames = list(raw_frames)
        low_level_stats = list(low_level_stats)

    # Extract V-JEPA latents
    sampled_frames = raw_frames[::STRIDE]
    all_latents = extract_all_latents(model, processor, sampled_frames)
    
    results = []
    num_steps = all_latents.shape[0]
    
    freeze_targets = False  # Set to True to freeze future frames


    # Sliding window evaluation
    for i in range(num_steps - max(HORIZONS) - CONTEXT_STEPS):
        context = all_latents[i : i + CONTEXT_STEPS]
        row = {"latent_step": i, "raw_frame_idx": i * TUBELET_SIZE * STRIDE}
        
        # Add low-level stats (aligned to end of context window)
        idx = min(row["raw_frame_idx"] + (CONTEXT_STEPS * TUBELET_SIZE * STRIDE), len(low_level_stats)-1)
        row.update(low_level_stats[idx])

        # Predict at multiple horizons
        for h in HORIZONS:
            pred = predict_at_horizon(model, context, h)
            target = all_latents[i + CONTEXT_STEPS : i + CONTEXT_STEPS + h]
            
            if freeze_targets:
                # Repeat the first target frame h times along the time dimension
                target = target[0].unsqueeze(0).repeat(h, 1, 1)

            # Only compute Euclidean distance
            eucl_error = compute_prediction_error(pred, target)
            row[f"unpredictability_h{h}_euclidean"] = eucl_error
            
        results.append(row)

    df = pd.DataFrame(results)
    
    # Compute horizon statistics
    horizon_stats = compute_horizon_stats(df)
    
    custom_suffix = f"frozen_{TUBELET_SIZE}t_{STRIDE}s_{HORIZONS[0]}{HORIZONS[1]}{HORIZONS[2]}_ctxt{CONTEXT_STEPS}_" if freeze_targets else f"_{TUBELET_SIZE}t_{STRIDE}s_{HORIZONS[0]}{HORIZONS[1]}{HORIZONS[2]}_ctxt{CONTEXT_STEPS}_"

    # Save results
    stem = video_path.stem
    csv_path = OUTPUT_DIR / f"{stem}_{frame_order}_{custom_suffix}metrics.csv"
    
    # Save dataframe with summary statistics appended
    with open(csv_path, 'w') as f:
        # Write header comment with summary stats
        f.write("# Summary Statistics\n")
        for stat_type, stat_vals in horizon_stats.items():
            for h, val in stat_vals.items():
                f.write(f"# {stat_type.upper()}_H{h}: {val:.6f}\n")
        f.write("#\n")
        # Write actual data
        df.to_csv(f, index=False)
    
    print(f"  ✓ Saved metrics to {csv_path.name}")
    
    # Print summary statistics
    print(f"  Summary Statistics:")
    print(f"    Mean Euclidean Distance:")
    for h, val in horizon_stats["mean"].items():
        print(f"      H{h}: {val:.6f}")
    print(f"    Max Euclidean Distance:")
    for h, val in horizon_stats["max"].items():
        print(f"      H{h}: {val:.6f}")
    
    # Compute and save correlation matrix
    corr_matrix = compute_correlation_matrix(df)
    corr_path = OUTPUT_DIR / f"{stem}_{frame_order}_{custom_suffix}correlation.csv"
    corr_matrix.to_csv(corr_path)
    print(f"  ✓ Saved correlation matrix to {corr_path.name}")
    
    # Create and save visualizations
    fig_metrics, horizon_stats = plot_metrics_over_time(df, stem, frame_order, horizon_stats)
    metrics_plot_path = OUTPUT_DIR / f"{stem}_{frame_order}_{custom_suffix}metrics.png"
    fig_metrics.savefig(metrics_plot_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig_metrics)
    print(f"  ✓ Saved metrics plot to {metrics_plot_path.name}")
    
    fig_corr = plot_correlation_matrix(corr_matrix, stem, frame_order)
    corr_plot_path = OUTPUT_DIR / f"{stem}_{frame_order}_{custom_suffix}correlation.png"
    fig_corr.savefig(corr_plot_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig_corr)
    print(f"  ✓ Saved correlation plot to {corr_plot_path.name}")
    
    return df, corr_matrix, horizon_stats

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()

    # Process videos
    video_files = list(VIDEO_DIR.glob("klepdonder.mp4"))
    
    for v_file in video_files:
        print(f"\n{'='*60}")
        print(f"Video: {v_file.name}")
        print(f"{'='*60}")
        
        for frame_order in ["forward", "backward", "shuffled"]:
            try:
                df, corr, stats = process_video(v_file, mdl, proc, frame_order=frame_order)
            except Exception as e:
                print(f"  ✗ Failed {v_file.name} ({frame_order}): {e}")
    
    print(f"\n{'='*60}")
    print(f"All processing complete! Results saved to {OUTPUT_DIR}")
    print(f"{'='*60}")
