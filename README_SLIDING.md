# V-JEPA2 Sliding Window Analysis

## Overview

This is an enhanced version of the original `vidpred_vjepa2.py` script that implements **frame-level predictability metrics** using a sliding window approach. Instead of analyzing a single middle segment of each video, the script now:

1. **Slides a window** across the entire video (e.g., frames 0-10, 1-11, 2-12, etc.)
2. **Predicts multiple future horizons** (4, 8, and 12 frames ahead by default)
3. **Computes visual statistics** for each frame window
4. **Outputs frame-level metrics** to a detailed CSV for further analysis

## Key Features

### Sliding Window Analysis
- Context window: 10 frames (configurable via `CONTEXT_TOKENS`)
- Future prediction: 4, 8, and 12 frames (configurable via `PREDICT_TOKENS_LIST`)
- Sliding stride: 1 frame (produces one analysis per frame in the video)
- Results: One row per window analyzed

### Visual Statistics Computed
For each context window, the script computes:

- **RMS Contrast** (`ctx_rms_contrast`): Contrast of the frame sequence (standard deviation of normalized pixel intensities)
- **Brightness** (`ctx_brightness`): Mean pixel intensity
- **Edge Content** (`ctx_edge_content`): Fraction of pixels detected as edges (via Canny edge detection)
- **Optical Flow Magnitude** (`ctx_optical_flow_magnitude`): Mean motion magnitude between consecutive frames

### Predictability Metrics
For each prediction horizon (4, 8, 12 frames), the script stores:

- `pred_N_cos_mean`: Mean cosine distance (latent space unpredictability)
- `pred_N_cos_std`: Standard deviation of cosine distance
- `pred_N_l2_mean`: Mean L2 distance
- `pred_N_l2_std`: Standard deviation of L2 distance

Where N ∈ {4, 8, 12}

## Output Files

Each video produces:
```
vjepa_results_sliding/
├── {video_name}/
│   └── frame_metrics.csv
```

The CSV contains:
- `frame_index`: Window starting position (0-based)
- Visual statistics (ctx_rms_contrast, ctx_brightness, etc.)
- Predictability metrics for each horizon (pred_4_cos_mean, pred_8_l2_mean, etc.)

## Configuration

Edit the top of `vidpred_vjepa2_sliding.py` to modify:

```python
HF_MODEL_NAME       = "facebook/vjepa2-vitg-fpc64-384"  # Model to use
CONTEXT_TOKENS      = 10                                 # Context window size (frames)
PREDICT_TOKENS_LIST = [4, 8, 12]                        # Future prediction horizons
TUBELET_SIZE        = 2                                  # Frame grouping
STRIDE              = 2                                  # Downsampling stride
OUTPUT_BASE_DIR     = Path("vjepa_results_sliding")     # Output directory
VIDEO_NAMES         = [...]                             # List of video names
```

## Usage

### Local Execution
```bash
# Activate environment
source /home/predatt/wiesche/generator_env/.venv/bin/activate

# Load CUDA (if needed)
module load cuda/11.4

# Run directly
python vidpred_vjepa2_sliding.py
```

### SLURM Submission (Recommended for HPC)

1. **Configure the SLURM script** (`submit_vjepa2_sliding.sh`):
   - Update `--mail-user` with your email
   - Adjust `--time` if needed (current: 12 hours for ~17 videos)
   - Modify `--mem` or `--cpus-per-task` based on your queue

2. **Submit the job**:
   ```bash
   sbatch submit_vjepa2_sliding.sh
   ```

3. **Monitor progress**:
   ```bash
   # View job status
   squeue -u $USER
   
   # View live output
   tail -f logs/vjepa2_sliding_*.out
   ```

4. **Check results**:
   ```bash
   # List generated CSVs
   find vjepa_results_sliding -name "*.csv"
   
   # Quick statistics
   head -5 vjepa_results_sliding/*/frame_metrics.csv
   ```

## Resource Requirements

**Current SLURM configuration** (`submit_vjepa2_sliding.sh`):
- **GPU**: 1 A100 (or similar GPU with 40GB+ VRAM)
- **CPU**: 8 cores (for frame I/O and preprocessing)
- **Memory**: 96 GB RAM
- **Time**: 12 hours (adjust as needed)
- **Partition**: gpu

**Estimated Runtimes** (for 17 videos, ~10k frames each):
- GPU time: 8-10 hours
- Wall time: 12 hours (with I/O, overhead)

## Output Example

```csv
frame_index,ctx_rms_contrast,ctx_brightness,ctx_edge_content,ctx_optical_flow_magnitude,pred_4_cos_mean,pred_4_cos_std,pred_4_l2_mean,pred_4_l2_std,pred_8_cos_mean,pred_8_cos_std,pred_8_l2_mean,pred_8_l2_std,pred_12_cos_mean,pred_12_cos_std,pred_12_l2_mean,pred_12_l2_std
0,0.2847,0.5623,0.0123,1.2456,0.3456,0.0234,52.123,3.456,0.4567,0.0345,58.234,4.567,0.5678,0.0456,64.345,5.678
1,0.2834,0.5634,0.0125,1.2389,0.3412,0.0231,51.890,3.423,0.4523,0.0342,57.901,4.534,0.5634,0.0453,64.012,5.645
...
```

## Tips

1. **Faster Iteration**: Start with fewer videos to test configuration
   ```python
   VIDEO_NAMES = ["deurinhuis"]  # Test with one video
   ```

2. **Long Videos**: The script processes all videos regardless of length, but very long videos (>100k frames) may require more RAM/time

3. **Visual Statistics**: To disable optical flow (faster but loses motion info), comment out the optical flow computation

4. **Reproducibility**: Results depend on model weights and may vary slightly with different CUDA versions

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or increase `--mem` in SLURM script
- **Model Loading Error**: Ensure internet connection for downloading model weights on first run
- **Slow Performance**: Check GPU utilization with `nvidia-smi`; may need to adjust `--cpus-per-task`

## Differences from Original Script

| Aspect | Original | Sliding Window |
|--------|----------|-----------------|
| Analysis Points | 1 (middle) | ~N-20 (all frames) |
| Predictions | Single horizon | Multiple (4, 8, 12 steps) |
| Visual Stats | None | 4 metrics per frame |
| Output | GIF + CSV | CSV only (frame-level) |
| Use Case | Quick overview | Detailed frame-level analysis |

## Citation

If using this script, consider citing:
- LeCun et al. 2024: V-JEPA2 model paper
- Your project/institution

---

**Last Updated**: March 2026
**Python Requirements**: torch, transformers, cv2, numpy, pandas, PIL
