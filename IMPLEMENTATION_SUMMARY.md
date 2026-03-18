# Sliding Window Analysis - Implementation Summary

## What's New

I've created three files to enable frame-level predictability analysis on your HPC:

### 1. **vidpred_vjepa2_sliding.py** (Main Analysis Script)

**Key Changes from Original:**

✅ **Sliding Window Architecture**
   - Iterates through video frame-by-frame (not just middle segment)
   - For each position: uses 10 context frames to predict 4, 8, or 12 frames ahead
   - Produces N-(context+max_predict) analysis windows per video
   - Example: 1000-frame video → ~968 windows analyzed

✅ **Multi-Horizon Predictions**
   - Original: Single prediction horizon
   - New: Simultaneous predictions for 4, 8, and 12 frames ahead
   - Captures different timescales of predictability
   - Configurable via `PREDICT_TOKENS_LIST = [4, 8, 12]`

✅ **Visual Statistics Per Frame**
   - **RMS Contrast**: Texture complexity (√(mean(I²)))
   - **Brightness**: Mean pixel intensity (0-1 scale)
   - **Edge Content**: Canny edge detection density
   - **Optical Flow**: Motion magnitude between consecutive frames
   - Computed for each context window; all stored in CSV

✅ **Single CSV Output Format**
   - One row per analyzed window
   - Columns: frame_index, visual stats, predictability metrics
   - Removed GIF generation (much faster)
   - Easy to load and analyze with pandas/numpy

✅ **Better Error Handling & Logging**
   - Progress indicators during analysis
   - Summary statistics printed after each video
   - Detailed error messages with traceback
   - Timestamp tracking for job monitoring

### 2. **submit_vjepa2_sliding.sh** (SLURM Submission Script)

**Features:**
- Standard SLURM preamble compatible with your cluster
- Auto-loads CUDA 11.4 module
- Activates your venv at the correct path
- Creates `logs/` directory automatically
- Prints diagnostic info (Python version, CUDA availability)
- 8 CPU cores for parallel frame I/O
- 96 GB RAM (for model + intermediate buffers)
- 1 A100 GPU (or compatible)
- 12-hour wall time (adjust for longer videos if needed)

**Usage:**
```bash
sbatch submit_vjepa2_sliding.sh
squeue -u $USER
tail -f logs/vjepa2_sliding_*.out
```

### 3. **README_SLIDING.md** (Documentation)

Complete guide including:
- Configuration options
- CSV output format explanation
- Resource requirements & estimated runtimes
- Troubleshooting
- Tips for optimization

---

## Example Workflow

### Step 1: Prepare
```bash
# Copy scripts to your HPC home
scp vidpred_vjepa2_sliding.py user@hpc:~/my_project/
scp submit_vjepa2_sliding.sh user@hpc:~/my_project/
```

### Step 2: Configure (Optional)
Edit `vidpred_vjepa2_sliding.py` if needed:
```python
CONTEXT_TOKENS = 10           # Change context window size
PREDICT_TOKENS_LIST = [4, 8]  # Shorter predictions for faster run
STRIDE = 1                     # Use every frame (vs. every 2nd frame)
```

### Step 3: Submit
```bash
cd ~/my_project
sbatch submit_vjepa2_sliding.sh

# Monitor
tail -f logs/vjepa2_sliding_*.out
```

### Step 4: Analyze Results
```bash
# Check what was generated
ls -la vjepa_results_sliding/*/frame_metrics.csv

# Load and analyze
python3 << 'EOF'
import pandas as pd

# Load results for one video
df = pd.read_csv('vjepa_results_sliding/deurinhuis/frame_metrics.csv')

# Check predictability trends
print(df[['frame_index', 'pred_4_cos_mean', 'pred_8_cos_mean', 'pred_12_cos_mean']].head(20))

# Correlation with visual stats
print(df[['ctx_rms_contrast', 'pred_4_cos_mean']].corr())
EOF
```

---

## CSV Output Columns Explained

```
frame_index                      Frame window starting position
ctx_rms_contrast                 Texture complexity of context (higher = more texture)
ctx_brightness                   Mean brightness (0-1)
ctx_edge_content                 Edge density (0-1, higher = more edges)
ctx_optical_flow_magnitude       Motion magnitude (pixels/frame)
pred_4_cos_mean                  Avg unpredictability 4 frames ahead
pred_4_cos_std                   Variance in prediction errors (4-frame)
pred_4_l2_mean                   L2 distance for 4-frame prediction
pred_4_l2_std                    L2 variance
[... same for pred_8_* and pred_12_* ...]
```

**Interpreting Metrics:**
- **Lower `pred_*_cos_mean`** = More predictable
- **Higher `ctx_optical_flow_magnitude`** = Fast motion (typically less predictable)
- **Higher `ctx_rms_contrast`** = Complex texture (may impact predictability)

---

## Computational Complexity

For a single video with N frames:

- **Windows analyzed**: ~(N - 40)
- **Per-window compute**:
  - V-JEPA latent extraction: ~50-100ms (GPU)
  - 3× prediction + metrics: ~30-50ms total
  - Visual stats: ~10ms (CPU)
  
**Example**: 
- 1000 frames → 960 windows
- ~50 frames/minute per GPU
- 1000 frames → ~20 min per video
- 17 videos → ~6 hours GPU + 2-3 hours overhead
- Total HPC time: ~10-12 hours (hence 12h wall time)

---

## Advanced Options

### Faster Execution
```python
PREDICT_TOKENS_LIST = [4, 8]     # Drop longest prediction
STRIDE = 2                         # Use every 2nd frame
CONTEXT_TOKENS = 5                # Shorter context (risky!)
```

### More Detail
```python
PREDICT_TOKENS_LIST = [2, 4, 8, 12, 16]  # More granularity
# Add to visual stats: temp. variance, histogram features, etc.
```

### Single Video Test
```python
VIDEO_NAMES = ["deurinhuis"]
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--mem` request seems wrong; actually increase if OOM |
| Job timeout | Increase `--time` to 16:00:00 |
| Model not downloading | First run needs internet; submit from login node first |
| Slow on GPU | Check `nvidia-smi` for other jobs; may be sharing GPU |
| CSV encoding issues | Files are UTF-8; load with pandas normally |

---

## Next Steps for Analysis

Once you have the CSVs, you can:

1. **Merge all videos**: `pd.concat([pd.read_csv(p) for p in glob('vjepa_results_sliding/*/frame_metrics.csv')])`

2. **Plot predictability vs. visual stats**:
   ```python
   import matplotlib.pyplot as plt
   df_merged = ...  # merged dataframe
   plt.scatter(df_merged['ctx_optical_flow_magnitude'], 
               df_merged['pred_8_cos_mean'], alpha=0.3)
   plt.xlabel('Optical Flow')
   plt.ylabel('Predictability (8-frame)')
   plt.show()
   ```

3. **Identify "critical frames"** with high unpredictability
4. **Correlate with behavioral data** if available

---

## Questions?

Refer to:
- **README_SLIDING.md** for detailed docs
- **Original script** for V-JEPA2 details
- **SLURM logs** (logs/vjepa2_sliding_*.out) for debug info

Good luck with your analysis! 🚀
