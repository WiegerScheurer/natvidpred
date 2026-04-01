"""
Analysis script for V-JEPA prediction error trajectories
=========================================================

This script reads the frame_metrics CSVs and creates visualizations to help understand
how prediction errors degrade across different prediction horizons.

Key questions it helps answer:
1. Do prediction errors monotonically increase with horizon?
2. How do errors differ at different frames within the prediction window?
3. Is max error more informative than mean error?
4. How do visual statistics correlate with prediction difficulty?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def analyze_error_degradation(csv_path, output_dir=None):
    """
    Analyze how prediction errors degrade across prediction horizons.
    
    Args:
        csv_path: Path to frame_metrics CSV
        output_dir: Directory to save plots (if None, uses same as CSV)
    """
    df = pd.read_csv(csv_path)
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract prediction step information
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    pred_steps_list = sorted(set(int(c.split('_')[1]) for c in pred_cols if '_' in c[5:]))
    
    print(f"\nAnalyzing: {csv_path}")
    print(f"Prediction horizons found: {pred_steps_list}")
    
    # --- Figure 1: Mean vs Max Error ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    means = []
    maxs = []
    stds = []
    
    for pred_steps in pred_steps_list:
        l2_mean_col = f'pred_{pred_steps}_l2_mean'
        l2_max_col = f'pred_{pred_steps}_l2_max'
        
        if l2_mean_col in df.columns:
            means.append(df[l2_mean_col].mean())
        if l2_max_col in df.columns:
            maxs.append(df[l2_max_col].mean())
        if l2_mean_col in df.columns:
            stds.append(df[l2_mean_col].std())
    
    # Plot mean and max
    axes[0].plot(pred_steps_list, means, 'o-', label='Mean Error', linewidth=2, markersize=8)
    if maxs:
        axes[0].plot(pred_steps_list, maxs, 's--', label='Max Error', linewidth=2, markersize=8)
    axes[0].fill_between(pred_steps_list, 
                          np.array(means) - np.array(stds), 
                          np.array(means) + np.array(stds), 
                          alpha=0.2)
    axes[0].set_xlabel('Prediction Steps', fontsize=12)
    axes[0].set_ylabel('L2 Distance', fontsize=12)
    axes[0].set_title('Prediction Error vs Horizon\n(Mean ± 1 SD)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # --- Figure 2: Per-frame error trajectory ---
    max_pred_steps = max(pred_steps_list)
    colors = plt.cm.viridis(np.linspace(0, 1, len(pred_steps_list)))
    
    for i, pred_steps in enumerate(pred_steps_list):
        per_frame_cols = [c for c in df.columns 
                         if c.startswith(f'pred_{pred_steps}_l2_frame_')]
        
        if per_frame_cols:
            per_frame_errors = []
            for frame_idx in range(pred_steps):
                col = f'pred_{pred_steps}_l2_frame_{frame_idx}'
                if col in df.columns:
                    per_frame_errors.append(df[col].mean())
            
            if per_frame_errors:
                axes[1].plot(range(len(per_frame_errors)), per_frame_errors, 
                           'o-', label=f'{pred_steps} steps', 
                           color=colors[i], linewidth=2, markersize=6)
    
    axes[1].set_xlabel('Frame Index Within Prediction Window', fontsize=12)
    axes[1].set_ylabel('L2 Distance', fontsize=12)
    axes[1].set_title('Error Trajectory Within Each Prediction Window', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"{Path(csv_path).stem}_error_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()
    
    # --- Figure 3: Correlation with visual statistics ---
    visual_stats = ['ctx_rms_contrast', 'ctx_brightness', 'ctx_edge_content', 'ctx_optical_flow_magnitude']
    available_stats = [s for s in visual_stats if s in df.columns]
    
    if available_stats and pred_steps_list:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, stat in enumerate(available_stats):
            ax = axes[idx]
            
            for pred_steps in pred_steps_list:
                l2_col = f'pred_{pred_steps}_l2_mean'
                if l2_col in df.columns:
                    ax.scatter(df[stat], df[l2_col], alpha=0.5, label=f'{pred_steps} steps')
            
            ax.set_xlabel(stat.replace('ctx_', '').replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel('L2 Distance', fontsize=11)
            ax.set_title(f'Prediction Error vs {stat.replace("ctx_", "").replace("_", " ").title()}', 
                        fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        corr_plot_path = output_dir / f"{Path(csv_path).stem}_correlations.png"
        plt.savefig(corr_plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved correlation plot to: {corr_plot_path}")
        plt.close()
    
    # --- Print detailed statistics ---
    print(f"\n{'='*60}")
    print(f"Error Degradation Analysis")
    print(f"{'='*60}")
    
    print(f"\nMean L2 Distance by Prediction Horizon:")
    for i, pred_steps in enumerate(pred_steps_list):
        l2_col = f'pred_{pred_steps}_l2_mean'
        l2_max_col = f'pred_{pred_steps}_l2_max'
        if l2_col in df.columns:
            mean_err = df[l2_col].mean()
            std_err = df[l2_col].std()
            print(f"  {pred_steps:2d} steps: {mean_err:7.4f} ± {std_err:.4f}", end="")
            if l2_max_col in df.columns:
                max_err = df[l2_max_col].mean()
                print(f"  (max: {max_err:.4f})")
            else:
                print()
    
    # Check if errors monotonically increase
    if len(means) > 1:
        diffs = np.diff(means)
        is_monotonic = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
        print(f"\nMonotonic degradation (expect: True): {is_monotonic}")
        if not is_monotonic:
            print("  ⚠️  WARNING: Errors do NOT monotonically increase!")
            print("     This suggests either:")
            print("     1. Ground truth encoding bias (fixed in v2.py)")
            print("     2. Model learns different dynamics at different horizons")
            print("     3. Latent space geometry is non-Euclidean")

    # Visual statistics summary
    print(f"\n{'='*60}")
    print(f"Visual Statistics Summary")
    print(f"{'='*60}")
    for stat in available_stats:
        print(f"\n{stat.replace('ctx_', '').replace('_', ' ').title()}:")
        print(f"  Mean: {df[stat].mean():.4f} ± {df[stat].std():.4f}")
        print(f"  Range: [{df[stat].min():.4f}, {df[stat].max():.4f}]")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze V-JEPA prediction error trajectories"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to frame_metrics CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save analysis plots (default: same as CSV)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.csv_path).exists():
        print(f"Error: File not found: {args.csv_path}")
        return
    
    analyze_error_degradation(args.csv_path, args.output_dir)

if __name__ == "__main__":
    main()
