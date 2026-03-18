import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
RESULTS_DIR = Path("vjepa_results_sliding")

# --- Helper Functions ---

def list_available_videos():
    """List all available video results."""
    videos = []
    if RESULTS_DIR.exists():
        for video_dir in sorted(RESULTS_DIR.iterdir()):
            if video_dir.is_dir():
                csv_path = video_dir / "frame_metrics.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    videos.append({
                        'name': video_dir.name,
                        'frames': len(df),
                        'path': csv_path
                    })
    return videos

def load_video(video_name):
    """Load metrics for a specific video."""
    csv_path = RESULTS_DIR / video_name / "frame_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

def get_visual_stats_cols(df):
    """Get visual statistics columns."""
    return sorted([col for col in df.columns if col.startswith('ctx_')])

def get_pred_cols(df, horizon=None):
    """Get prediction columns."""
    pred_cols = [col for col in df.columns if col.startswith('pred_') and '_mean' in col]
    if horizon:
        pred_cols = [col for col in pred_cols if f'_{horizon}_' in col]
    return sorted(pred_cols)

# --- Interactive Functions ---

def quick_compare(video_name, metric1, metric2, output_path=None):
    """Quick comparison of two metrics."""
    df = load_video(video_name)
    if df is None:
        print(f"Video '{video_name}' not found!")
        return
    
    if metric1 not in df.columns or metric2 not in df.columns:
        print(f"Metric not found in data!")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series
    frame_idx = df['frame_index'].values
    ax = axes[0]
    ax.plot(frame_idx, df[metric1], label=metric1, linewidth=2, alpha=0.8)
    ax_twin = ax.twinx()
    ax_twin.plot(frame_idx, df[metric2], label=metric2, color='red', linewidth=2, alpha=0.8)
    ax.set_xlabel('Frame Index', fontweight='bold')
    ax.set_ylabel(metric1, fontweight='bold')
    ax_twin.set_ylabel(metric2, color='red', fontweight='bold')
    ax.set_title(f'{video_name}: {metric1} vs {metric2}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Scatter
    ax = axes[1]
    clean_df = df[[metric1, metric2]].dropna()
    scatter = ax.scatter(clean_df[metric1], clean_df[metric2], 
                        c=frame_idx[:len(clean_df)], cmap='viridis', alpha=0.6, s=20)
    
    # Trend line
    z = np.polyfit(clean_df[metric1], clean_df[metric2], 1)
    p = np.poly1d(z)
    x_line = np.linspace(clean_df[metric1].min(), clean_df[metric1].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    corr, pval = pearsonr(clean_df[metric1], clean_df[metric2])
    ax.set_xlabel(metric1, fontweight='bold')
    ax.set_ylabel(metric2, fontweight='bold')
    ax.set_title(f'Correlation: r={corr:.3f} (p={pval:.2e})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Frame Index', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.show()

def extract_window(video_name, start_frame, end_frame, output_path=None):
    """Extract and plot a specific window from a video."""
    df = load_video(video_name)
    if df is None:
        print(f"Video '{video_name}' not found!")
        return
    
    window_data = df[(df['frame_index'] >= start_frame) & (df['frame_index'] <= end_frame)]
    
    if window_data.empty:
        print(f"No data in frame range {start_frame}-{end_frame}")
        return
    
    visual_cols = get_visual_stats_cols(df)
    pred_cols = get_pred_cols(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    frame_range = window_data['frame_index'].values
    
    # Visual statistics
    ax = axes[0, 0]
    for col in visual_cols:
        ax.plot(frame_range, window_data[col], label=col.replace('ctx_', ''), 
               linewidth=2, marker='o', markersize=4, alpha=0.8)
    ax.set_ylabel('Visual Metrics', fontweight='bold')
    ax.set_title('Visual Statistics', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Optical flow
    ax = axes[0, 1]
    ax.plot(frame_range, window_data['ctx_optical_flow_magnitude'], 
           color='orangered', linewidth=2.5, marker='s', markersize=5)
    ax.fill_between(frame_range, window_data['ctx_optical_flow_magnitude'], alpha=0.3, color='orangered')
    ax.set_ylabel('Optical Flow', fontweight='bold', color='orangered')
    ax.set_title('Motion Activity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Cosine distances
    ax = axes[1, 0]
    cos_cols = [col for col in pred_cols if 'cos' in col]
    for col in cos_cols:
        ax.plot(frame_range, window_data[col], 
               label=col.replace('pred_', '').replace('_cos_mean', ''), 
               linewidth=2, marker='D', markersize=4, alpha=0.8)
    ax.set_ylabel('Cosine Distance', fontweight='bold')
    ax.set_title('Unpredictability by Horizon', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # L2 distances
    ax = axes[1, 1]
    l2_cols = [col for col in pred_cols if 'l2' in col]
    for col in l2_cols:
        ax.plot(frame_range, window_data[col], 
               label=col.replace('pred_', '').replace('_l2_mean', ''), 
               linewidth=2, marker='^', markersize=4, alpha=0.8, linestyle='--')
    ax.set_xlabel('Frame Index', fontweight='bold')
    ax.set_ylabel('L2 Distance', fontweight='bold')
    ax.set_title('L2 Distance by Horizon', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{video_name}: Frames {start_frame}-{end_frame}', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.show()

def find_interesting_frames(video_name, metric='pred_8_cos_mean', threshold='top', n=10):
    """Find and display frames with unusual metric values."""
    df = load_video(video_name)
    if df is None:
        print(f"Video '{video_name}' not found!")
        return
    
    if metric not in df.columns:
        print(f"Metric '{metric}' not found!")
        return
    
    clean_df = df[['frame_index', metric]].dropna()
    
    if threshold == 'top':
        interesting = clean_df.nlargest(n, metric)
        title_prefix = f"Most Unpredictable"
    elif threshold == 'bottom':
        interesting = clean_df.nsmallest(n, metric)
        title_prefix = f"Most Predictable"
    else:
        print("threshold must be 'top' or 'bottom'")
        return
    
    print(f"\n{title_prefix} frames in {video_name}:")
    print(interesting.to_string(index=False))
    print()
    
    # Visualize these frames in context
    fig, ax = plt.subplots(figsize=(14, 5))
    
    frame_idx = df['frame_index'].values
    metric_vals = df[metric].values
    
    ax.plot(frame_idx, metric_vals, 'o-', linewidth=1, markersize=3, alpha=0.5, label='All frames')
    
    interesting_frames = interesting['frame_index'].values
    interesting_vals = interesting[metric].values
    
    if threshold == 'top':
        ax.scatter(interesting_frames, interesting_vals, s=200, marker='*', 
                  color='red', edgecolor='darkred', linewidth=1.5, label=f'Top {n} {threshold}', zorder=5)
    else:
        ax.scatter(interesting_frames, interesting_vals, s=200, marker='*', 
                  color='green', edgecolor='darkgreen', linewidth=1.5, label=f'Top {n} {threshold}', zorder=5)
    
    ax.set_xlabel('Frame Index', fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'{video_name}: {title_prefix} Frames (metric: {metric})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return interesting

def correlation_with_metric(video_name, target_metric, output_path=None):
    """Show correlations of all metrics with a target metric."""
    df = load_video(video_name)
    if df is None:
        print(f"Video '{video_name}' not found!")
        return
    
    if target_metric not in df.columns:
        print(f"Target metric '{target_metric}' not found!")
        return
    
    visual_cols = get_visual_stats_cols(df)
    pred_cols = get_pred_cols(df)
    all_cols = visual_cols + pred_cols
    
    correlations = {}
    for col in all_cols:
        if col != target_metric:
            clean_data = df[[col, target_metric]].dropna()
            if len(clean_data) > 1:
                corr, pval = pearsonr(clean_data[col], clean_data[target_metric])
                correlations[col] = {'r': corr, 'p': pval}
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]['r']), reverse=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics = [item[0].replace('ctx_', '').replace('pred_', '').replace('_mean', '') for item in sorted_corrs]
    corrs = [item[1]['r'] for item in sorted_corrs]
    colors = ['green' if c > 0 else 'red' for c in corrs]
    
    ax.barh(metrics, corrs, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Pearson Correlation Coefficient', fontweight='bold')
    ax.set_title(f'{video_name}: Correlations with {target_metric}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add p-value annotations
    for i, (metric, corr_val) in enumerate(zip(metrics, corrs)):
        pval = sorted_corrs[i][1]['p']
        sig = '**' if pval < 0.001 else '*' if pval < 0.05 else ''
        ax.text(corr_val + 0.01 if corr_val > 0 else corr_val - 0.01, i, 
               f'{corr_val:.2f}{sig}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.show()
    
    # Print table
    print(f"\nCorrelations with {target_metric}:")
    print("-" * 60)
    print(f"{'Metric':<40} {'r':<10} {'p-value':<15}")
    print("-" * 60)
    for metric, corr_data in sorted_corrs:
        metric_name = metric.replace('ctx_', '').replace('pred_', '').replace('_mean', '')
        print(f"{metric_name:<40} {corr_data['r']:>9.3f} {corr_data['p']:>14.2e}")

# --- CLI Interface ---

def print_menu():
    """Print interactive menu."""
    print(f"\n{'='*60}")
    print("V-JEPA2 Interactive Analysis Tool")
    print(f"{'='*60}")
    print("\nAvailable commands:")
    print("  1. List videos")
    print("  2. Quick compare (two metrics)")
    print("  3. Extract window (specific frame range)")
    print("  4. Find interesting frames")
    print("  5. Correlation analysis")
    print("  0. Exit")
    print()

def interactive_mode():
    """Run in interactive mode."""
    videos = list_available_videos()
    
    if not videos:
        print("No video results found!")
        return
    
    print(f"\nFound {len(videos)} videos")
    for i, v in enumerate(videos):
        print(f"  [{i}] {v['name']} ({v['frames']} frames)")
    
    while True:
        print_menu()
        choice = input("Enter command (0-5): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        
        elif choice == '1':
            print(f"\nAvailable videos:")
            for i, v in enumerate(videos):
                print(f"  [{i}] {v['name']} ({v['frames']} frames)")
        
        elif choice == '2':
            video_idx = int(input(f"Select video [0-{len(videos)-1}]: "))
            video_name = videos[video_idx]['name']
            
            df = load_video(video_name)
            print("\nAvailable metrics:")
            all_cols = get_visual_stats_cols(df) + get_pred_cols(df)
            for i, col in enumerate(all_cols):
                print(f"  [{i}] {col}")
            
            m1_idx = int(input("Select metric 1 index: "))
            m2_idx = int(input("Select metric 2 index: "))
            
            metric1 = all_cols[m1_idx]
            metric2 = all_cols[m2_idx]
            
            quick_compare(video_name, metric1, metric2)
        
        elif choice == '3':
            video_idx = int(input(f"Select video [0-{len(videos)-1}]: "))
            video_name = videos[video_idx]['name']
            max_frames = videos[video_idx]['frames']
            
            start = int(input(f"Start frame [0-{max_frames-1}]: "))
            end = int(input(f"End frame [0-{max_frames-1}]: "))
            
            extract_window(video_name, start, end)
        
        elif choice == '4':
            video_idx = int(input(f"Select video [0-{len(videos)-1}]: "))
            video_name = videos[video_idx]['name']
            
            df = load_video(video_name)
            pred_cols = get_pred_cols(df)
            print("\nAvailable metrics for outlier detection:")
            for i, col in enumerate(pred_cols):
                print(f"  [{i}] {col}")
            
            metric_idx = int(input("Select metric: "))
            metric = pred_cols[metric_idx]
            
            threshold = input("Top (high values) or bottom (low values)? [top/bottom]: ")
            n = int(input("How many frames? [default: 10]: ") or "10")
            
            find_interesting_frames(video_name, metric, threshold, n)
        
        elif choice == '5':
            video_idx = int(input(f"Select video [0-{len(videos)-1}]: "))
            video_name = videos[video_idx]['name']
            
            df = load_video(video_name)
            all_cols = get_visual_stats_cols(df) + get_pred_cols(df)
            print("\nAvailable metrics:")
            for i, col in enumerate(all_cols):
                print(f"  [{i}] {col}")
            
            target_idx = int(input("Select target metric: "))
            target_metric = all_cols[target_idx]
            
            correlation_with_metric(video_name, target_metric)
        
        else:
            print("Invalid command!")

if __name__ == "__main__":
    interactive_mode()
