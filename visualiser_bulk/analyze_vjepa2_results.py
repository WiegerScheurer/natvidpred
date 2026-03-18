import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
RESULTS_DIR = Path("/project/3018078.02/natvidpred_workspace/vjepa_results_sliding")
OUTPUT_FIGS_DIR = Path("/project/3018078.02/natvidpred_workspace/analysis_figures")
WINDOW_SIZE = 100  # Frame window for detailed plots
SMOOTH_WINDOW = 5  # Rolling average window

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- Data Loading & Preparation ---

def load_video_metrics(video_name):
    """Load metrics CSV for a specific video."""
    csv_path = RESULTS_DIR / video_name / "frame_metrics.csv"
    if not csv_path.exists():
        print(f"Warning: CSV not found for {video_name}")
        return None
    return pd.read_csv(csv_path)

def load_all_videos():
    """Load all available video metrics."""
    videos = {}
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return videos
    
    for video_dir in sorted(RESULTS_DIR.iterdir()):
        if video_dir.is_dir():
            csv_path = video_dir / "frame_metrics.csv"
            if csv_path.exists():
                videos[video_dir.name] = pd.read_csv(csv_path)
                print(f"Loaded: {video_dir.name} ({len(videos[video_dir.name])} frames)")
    
    return videos

def get_visual_stats_cols(df):
    """Extract visual statistics column names from dataframe."""
    return [col for col in df.columns if col.startswith('ctx_')]

def get_pred_cols(df, horizon=None):
    """Extract prediction columns, optionally filtered by horizon."""
    pred_cols = [col for col in df.columns if col.startswith('pred_')]
    
    if horizon:
        pred_cols = [col for col in pred_cols if f'_{horizon}_' in col]
    
    return pred_cols

# --- Analysis Functions ---

def compute_correlations(df):
    """Compute correlations between all metrics."""
    visual_cols = get_visual_stats_cols(df)
    pred_cols = get_pred_cols(df)
    
    # Focus on mean predictability metrics (not std)
    pred_cols = [col for col in pred_cols if '_mean' in col]
    
    all_cols = visual_cols + pred_cols
    corr_matrix = df[all_cols].corr()
    
    return corr_matrix, visual_cols, pred_cols

def compute_rolling_correlation(df, window=20):
    """Compute rolling correlation between optical flow and unpredictability."""
    visual_cols = get_visual_stats_cols(df)
    pred_cols = [col for col in get_pred_cols(df) if '_mean' in col]
    
    rolling_corrs = {}
    
    for pred_col in pred_cols:
        rolling_corrs[pred_col] = []
        for i in range(len(df) - window):
            window_data = df.iloc[i:i+window]
            corr, _ = pearsonr(window_data['ctx_optical_flow_magnitude'], window_data[pred_col])
            rolling_corrs[pred_col].append(corr)
    
    return rolling_corrs

# --- Visualization Functions ---

def plot_correlation_matrix(df, video_name):
    """Create correlation matrix heatmap."""
    corr_matrix, visual_cols, pred_cols = compute_correlations(df)
    
    # Reorder: visual stats first, then predictions
    ordered_cols = visual_cols + pred_cols
    corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, 
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(f'Metric Correlations - {video_name}', fontsize=14, fontweight='bold')
    
    # Improve labels
    labels = [label.get_text().replace('ctx_', '').replace('pred_', 'pred ').replace('_mean', '') 
              for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout()
    return fig

def plot_timeseries_full(df, video_name):
    """Plot all metrics as time series (full video)."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    visual_cols = get_visual_stats_cols(df)
    pred_cols = sorted([col for col in get_pred_cols(df) if '_mean' in col])
    
    frame_idx = df['frame_index'].values
    
    # Visual statistics
    for col in visual_cols:
        axes[0].plot(frame_idx, df[col], label=col.replace('ctx_', ''), linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Visual Statistics', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Full Video Time Series - {video_name}', fontsize=13, fontweight='bold')
    
    # Optical flow
    axes[1].plot(frame_idx, df['ctx_optical_flow_magnitude'], color='orangered', linewidth=1.5, label='Optical Flow')
    axes[1].fill_between(frame_idx, df['ctx_optical_flow_magnitude'], alpha=0.3, color='orangered')
    axes[1].set_ylabel('Optical Flow Magnitude', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Predictability metrics - cosine distance
    cos_cols = [col for col in pred_cols if 'cos' in col]
    for col in cos_cols:
        axes[2].plot(frame_idx, df[col], label=col.replace('pred_', 'Pred ').replace('_cos_mean', ''), 
                    linewidth=1.5, alpha=0.8)
    axes[2].set_ylabel('Cosine Distance (Unpredictability)', fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    # Predictability metrics - L2 distance
    l2_cols = [col for col in pred_cols if 'l2' in col]
    for col in l2_cols:
        axes[3].plot(frame_idx, df[col], label=col.replace('pred_', 'Pred ').replace('_l2_mean', ''), 
                    linewidth=1.5, alpha=0.8, linestyle='--')
    axes[3].set_ylabel('L2 Distance', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    axes[3].legend(loc='upper right', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_windowed_timeseries(df, video_name, window_size=None, num_windows=3):
    """Plot detailed windowed views of metrics."""
    if window_size is None:
        window_size = WINDOW_SIZE
    
    # Select window start positions
    max_start = max(0, len(df) - window_size)
    if num_windows > 1:
        window_starts = np.linspace(0, max_start, num_windows, dtype=int)
    else:
        window_starts = [0]
    
    visual_cols = get_visual_stats_cols(df)
    pred_cols = sorted([col for col in get_pred_cols(df) if '_mean' in col])
    cos_cols = [col for col in pred_cols if 'cos' in col]
    
    fig, axes = plt.subplots(num_windows, 3, figsize=(18, 4*num_windows))
    if num_windows == 1:
        axes = axes.reshape(1, -1)
    
    for window_idx, start in enumerate(window_starts):
        end = min(start + window_size, len(df))
        window_data = df.iloc[start:end].copy()
        frame_range = window_data['frame_index'].values
        
        # Column 1: Visual statistics
        ax = axes[window_idx, 0]
        for col in visual_cols:
            ax.plot(frame_range, window_data[col], label=col.replace('ctx_', ''), 
                   linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax.set_ylabel('Visual Statistics', fontweight='bold')
        ax.set_title(f'Visual Stats (Frames {start}-{end})', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Column 2: Motion and unpredictability
        ax = axes[window_idx, 1]
        ax2 = ax.twinx()
        
        line1 = ax.plot(frame_range, window_data['ctx_optical_flow_magnitude'], 
                       color='orangered', linewidth=2.5, marker='s', label='Optical Flow', alpha=0.8)
        line2 = ax2.plot(frame_range, window_data['pred_8_cos_mean'], 
                        color='steelblue', linewidth=2.5, marker='o', label='Unpredictability', alpha=0.8)
        
        ax.set_ylabel('Optical Flow', color='orangered', fontweight='bold')
        ax2.set_ylabel('Cosine Distance', color='steelblue', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='orangered')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax.set_title(f'Motion vs Unpredictability (Frames {start}-{end})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Column 3: Prediction horizons
        ax = axes[window_idx, 2]
        for col in cos_cols:
            ax.plot(frame_range, window_data[col], 
                   label=col.replace('pred_', '').replace('_cos_mean', ' steps'), 
                   linewidth=2, marker='D', markersize=4, alpha=0.8)
        ax.set_ylabel('Cosine Distance', fontweight='bold')
        ax.set_title(f'Prediction Horizons (Frames {start}-{end})', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('Frame Index', fontweight='bold')
    
    fig.suptitle(f'Windowed Analysis - {video_name}', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig

def plot_scatter_relationships(df, video_name):
    """Create scatter plots showing key relationships."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Select key metrics
    visual_cols = get_visual_stats_cols(df)
    pred_8_cos = 'pred_8_cos_mean'
    pred_4_cos = 'pred_4_cos_mean'
    
    # Remove NaN values
    clean_df = df[visual_cols + [pred_8_cos, pred_4_cos]].dropna()
    
    relationships = [
        ('ctx_optical_flow_magnitude', pred_8_cos, 'Motion vs Unpredictability (8-step)'),
        ('ctx_rms_contrast', pred_8_cos, 'Contrast vs Unpredictability (8-step)'),
        ('ctx_brightness', pred_8_cos, 'Brightness vs Unpredictability (8-step)'),
        ('ctx_edge_content', pred_8_cos, 'Edge Content vs Unpredictability (8-step)'),
        ('ctx_optical_flow_magnitude', pred_4_cos, 'Motion vs Unpredictability (4-step)'),
        ('ctx_rms_contrast', pred_4_cos, 'Contrast vs Unpredictability (4-step)'),
    ]
    
    for idx, (x_col, y_col, title) in enumerate(relationships):
        ax = axes[idx // 3, idx % 3]
        
        x = clean_df[x_col]
        y = clean_df[y_col]
        
        # Scatter plot with density coloring
        scatter = ax.scatter(x, y, c=range(len(x)), cmap='viridis', alpha=0.6, s=20)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        # Compute correlation
        corr, pval = pearsonr(x, y)
        
        ax.set_xlabel(x_col.replace('ctx_', ''), fontweight='bold')
        ax.set_ylabel(y_col.replace('pred_', 'Pred ').replace('_mean', ''), fontweight='bold')
        ax.set_title(f'{title}\n(r={corr:.2f}, p<0.001)' if pval < 0.001 else f'{title}\n(r={corr:.2f}, p={pval:.3f})', 
                    fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    fig.suptitle(f'Relationship Analysis - {video_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_rolling_correlation(df, video_name, window=20):
    """Plot rolling correlation between optical flow and unpredictability."""
    rolling_corrs = compute_rolling_correlation(df, window=window)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    frame_range = df['frame_index'].values[:len(df) - window]
    
    for pred_col, corr_values in rolling_corrs.items():
        horizon = pred_col.split('_')[1]
        ax.plot(frame_range, corr_values, label=f'{horizon}-step', linewidth=2, alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(frame_range, 0, 1, alpha=0.1, color='green', label='Positive correlation')
    ax.fill_between(frame_range, -1, 0, alpha=0.1, color='red', label='Negative correlation')
    
    ax.set_xlabel('Frame Index', fontweight='bold', fontsize=11)
    ax.set_ylabel('Pearson Correlation Coefficient', fontweight='bold', fontsize=11)
    ax.set_title(f'Rolling Correlation: Optical Flow vs Unpredictability\n(Window size: {window} frames) - {video_name}', 
                fontsize=12, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_metric_distributions(df, video_name):
    """Plot distributions of all metrics."""
    visual_cols = get_visual_stats_cols(df)
    pred_cols = sorted([col for col in get_pred_cols(df) if '_mean' in col and 'cos' in col])
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Visual statistics distributions
    for idx, col in enumerate(visual_cols):
        ax = axes[0, idx]
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(col.replace('ctx_', ''), fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{col.replace("ctx_", "")}\n(μ={data.mean():.3f}, σ={data.std():.3f})', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Predictability distributions
    for idx, col in enumerate(pred_cols):
        ax = axes[1, idx]
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax.set_xlabel(col.replace('pred_', 'Pred ').replace('_cos_mean', ''), fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{col.replace("pred_", "Pred ").replace("_cos_mean", "")}\n(μ={data.mean():.3f}, σ={data.std():.3f})', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Metric Distributions - {video_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_complexity_vs_predictability(df, video_name):
    """Create a synthetic 'complexity score' and relate to predictability."""
    # Normalize metrics
    scaler = StandardScaler()
    visual_cols = get_visual_stats_cols(df)
    
    visual_data = df[visual_cols].copy()
    visual_normalized = scaler.fit_transform(visual_data.fillna(visual_data.mean()))
    
    # Complexity = mean of all visual metrics (higher = more complex)
    complexity = np.mean(np.abs(visual_normalized), axis=1)
    
    pred_8_cos = df['pred_8_cos_mean'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series
    frame_idx = df['frame_index'].values
    axes[0].fill_between(frame_idx, complexity, alpha=0.3, color='purple', label='Complexity Score')
    ax_pred = axes[0].twinx()
    ax_pred.plot(frame_idx, pred_8_cos, color='red', linewidth=2, label='Unpredictability', alpha=0.8)
    
    axes[0].set_xlabel('Frame Index', fontweight='bold')
    axes[0].set_ylabel('Complexity Score (Normalized)', fontweight='bold', color='purple')
    ax_pred.set_ylabel('Cosine Distance', fontweight='bold', color='red')
    axes[0].set_title('Complexity vs Unpredictability Over Time', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='y', labelcolor='purple')
    ax_pred.tick_params(axis='y', labelcolor='red')
    
    # Scatter
    scatter = axes[1].scatter(complexity, pred_8_cos, c=frame_idx, cmap='cool', alpha=0.6, s=30)
    z = np.polyfit(complexity, pred_8_cos, 1)
    p = np.poly1d(z)
    x_line = np.linspace(complexity.min(), complexity.max(), 100)
    axes[1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    corr, pval = pearsonr(complexity, pred_8_cos)
    axes[1].set_xlabel('Complexity Score (Normalized)', fontweight='bold')
    axes[1].set_ylabel('Unpredictability (8-step)', fontweight='bold')
    axes[1].set_title(f'Complexity-Predictability Relationship\n(r={corr:.2f})', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Frame Index', fontweight='bold')
    
    fig.suptitle(f'Scene Complexity Analysis - {video_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# --- Report Generation ---

def generate_report(video_name):
    """Generate comprehensive analysis report for a video."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {video_name}")
    print(f"{'='*60}")
    
    df = load_video_metrics(video_name)
    if df is None:
        return
    
    # Create figure directory
    video_fig_dir = OUTPUT_FIGS_DIR / video_name
    video_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation matrix
    print("Generating correlation matrix...")
    fig = plot_correlation_matrix(df, video_name)
    fig.savefig(video_fig_dir / "01_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Full time series
    print("Generating full time series...")
    fig = plot_timeseries_full(df, video_name)
    fig.savefig(video_fig_dir / "02_timeseries_full.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Windowed analysis
    print("Generating windowed analysis...")
    fig = plot_windowed_timeseries(df, video_name, num_windows=3)
    fig.savefig(video_fig_dir / "03_windowed_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Scatter relationships
    print("Generating scatter relationships...")
    fig = plot_scatter_relationships(df, video_name)
    fig.savefig(video_fig_dir / "04_scatter_relationships.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Rolling correlation
    print("Generating rolling correlation...")
    fig = plot_rolling_correlation(df, video_name, window=20)
    fig.savefig(video_fig_dir / "05_rolling_correlation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 6. Distributions
    print("Generating distributions...")
    fig = plot_metric_distributions(df, video_name)
    fig.savefig(video_fig_dir / "06_distributions.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 7. Complexity vs predictability
    print("Generating complexity analysis...")
    fig = plot_complexity_vs_predictability(df, video_name)
    fig.savefig(video_fig_dir / "07_complexity_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Print statistics
    print_statistics(df, video_name, video_fig_dir)
    
    print(f"✓ Figures saved to: {video_fig_dir}")

def print_statistics(df, video_name, output_dir):
    """Print and save detailed statistics."""
    stats_text = f"\n{'='*60}\nSTATISTICS FOR {video_name}\n{'='*60}\n\n"
    
    # Visual statistics
    stats_text += "VISUAL STATISTICS:\n"
    visual_cols = get_visual_stats_cols(df)
    for col in visual_cols:
        data = df[col].dropna()
        stats_text += f"  {col}:\n"
        stats_text += f"    Mean: {data.mean():.4f}, Std: {data.std():.4f}\n"
        stats_text += f"    Min: {data.min():.4f}, Max: {data.max():.4f}\n"
    
    # Predictability metrics
    stats_text += "\nPREDICTABILITY METRICS:\n"
    pred_cols = sorted([col for col in get_pred_cols(df) if '_mean' in col])
    for col in pred_cols:
        data = df[col].dropna()
        stats_text += f"  {col}:\n"
        stats_text += f"    Mean: {data.mean():.4f}, Std: {data.std():.4f}\n"
        stats_text += f"    Min: {data.min():.4f}, Max: {data.max():.4f}\n"
    
    # Key correlations
    stats_text += "\nKEY CORRELATIONS (with 8-step unpredictability):\n"
    pred_8_cos = 'pred_8_cos_mean'
    for col in visual_cols:
        clean_data = df[[col, pred_8_cos]].dropna()
        if len(clean_data) > 0:
            corr, pval = pearsonr(clean_data[col], clean_data[pred_8_cos])
            stats_text += f"  {col} -> {pred_8_cos}: r={corr:.3f} (p={pval:.2e})\n"
    
    print(stats_text)
    
    # Save to file
    stats_file = output_dir / "statistics.txt"
    with open(stats_file, 'w') as f:
        f.write(stats_text)

def generate_summary_report(all_videos):
    """Generate cross-video comparison summary."""
    if not all_videos:
        print("No videos loaded!")
        return
    
    summary_fig_dir = OUTPUT_FIGS_DIR / "_summary"
    summary_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile statistics across all videos
    all_stats = []
    for video_name, df in all_videos.items():
        pred_cols = [col for col in get_pred_cols(df) if '_mean' in col and 'cos' in col]
        visual_cols = get_visual_stats_cols(df)
        
        stats_dict = {'video': video_name}
        
        # Mean metrics
        for col in visual_cols + pred_cols:
            stats_dict[col] = df[col].mean()
        
        all_stats.append(stats_dict)
    
    if not all_stats:
        return
    
    stats_df = pd.DataFrame(all_stats)
    
    # Plot cross-video comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Visual metrics comparison
    visual_cols = get_visual_stats_cols(all_videos[list(all_videos.keys())[0]])
    stats_df_visual = stats_df.set_index('video')[visual_cols]
    stats_df_visual.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title('Visual Statistics Across Videos', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Mean Value (Normalized)', fontweight='bold')
    axes[0].legend(title='Metric', fontsize=8)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Predictability metrics comparison
    pred_cols = [col for col in stats_df.columns if col.startswith('pred_')]
    stats_df_pred = stats_df.set_index('video')[pred_cols]
    stats_df_pred.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_title('Predictability Metrics Across Videos', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Mean Cosine Distance', fontweight='bold')
    axes[1].legend(title='Prediction Horizon', fontsize=8)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Cross-Video Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(summary_fig_dir / "cross_video_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save comparison data
    stats_df.to_csv(summary_fig_dir / "video_statistics_summary.csv", index=False)
    
    print(f"\n✓ Summary report saved to: {summary_fig_dir}")
    print(f"✓ Statistics saved to: {summary_fig_dir / 'video_statistics_summary.csv'}")

# --- Main ---

def main():
    """Main execution."""
    print(f"\n{'='*60}")
    print("V-JEPA2 Sliding Window Analysis Visualization")
    print(f"{'='*60}")
    
    # Load all videos
    all_videos = load_all_videos()
    
    if not all_videos:
        print("No videos found in results directory!")
        return
    
    print(f"\nLoaded {len(all_videos)} videos\n")
    
    # Generate reports for each video
    for video_name in sorted(all_videos.keys()):
        try:
            generate_report(video_name)
        except Exception as e:
            print(f"✗ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    try:
        generate_summary_report(all_videos)
    except Exception as e:
        print(f"✗ Error generating summary: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {OUTPUT_FIGS_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
