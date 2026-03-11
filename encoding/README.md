# MEG Visual Encoding Analysis

Fits a **temporal response function (TRF)** via ridge regression to predict 
MEG sensor signals from per-frame visual features (RMS luminance, SSIM motion proxy)
extracted from the stimulus videos. Includes k-fold cross-validation and a suite
of sanity-check figures.

---

## Quick start

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate meg_encoding
```

### 2. Configure paths in the SLURM script

Edit `run_encoding_analysis.sh`:

```bash
PROJECT_DIR="/path/to/project"   # root of your data
DATA_DIR="${PROJECT_DIR}/mat_files"
VIDEO_DIR="${PROJECT_DIR}/videos"
CONDITION_TABLE="${PROJECT_DIR}/ConditionTable.csv"
```

The script expects:
- `DATA_DIR/sub001_100Hz_badmuscle_badlowfreq_badcomp.mat`  (or `_v7_3.mat`)
- `VIDEO_DIR/ProjectAttention_movie_part41_24Hz.mp4`
- `VIDEO_DIR/ProjectAttention_movie_part41_bw_24Hz.mp4`  (backward conditions)

### 3. Submit

```bash
# Single subject (subject 1, conditions 1 & 3 by default)
sbatch run_encoding_analysis.sh

# Or array over all subjects (edit --array in the .sh file first)
sbatch --array=1-72 run_encoding_analysis.sh

# Local test run
python meg_encoding_analysis.py \
    --subject 1 \
    --conditions 1 3 \
    --data_dir /path/to/mat_files \
    --video_dir /path/to/videos \
    --condition_table /path/to/ConditionTable.csv \
    --output_dir ./output_sub001
```

---

## Conditions

| Code | Label            | Video variant       |
|------|------------------|---------------------|
| 1    | Attend forward   | `part<NN>_24Hz.mp4` |
| 2    | Unattend forward | `part<NN>_24Hz.mp4` |
| 3    | Attend backward  | `part<NN>_bw_24Hz.mp4` |
| 4    | Unattend backward| `part<NN>_bw_24Hz.mp4` |

Change `CONDITIONS="1 3"` in the SLURM script (or `--conditions` CLI arg) to
run any combination.

---

## MAT file structure assumption

The script assumes a **FieldTrip** continuous data structure with:
- `data.trial`   — cell array of (n_channels × n_samples) matrices, one per run
- `data.time`    — cell array of time vectors
- `data.label`   — channel labels
- `data.fsample` — sampling frequency (100 Hz)

Trial index is assumed to equal `run_number − 1` (i.e. run 1 → trial[0]).

**If your indexing differs**, adjust `run_trial_map` in `main()`:
```python
run_trial_map = {run_no: run_no - 1 for run_no, _, _ in runs}
```

The script prints the full MAT/HDF5 structure at INFO level so you can inspect it.

---

## Model

```
Visual features (RMS, SSIM)  [24 fps]
         │
         ▼  resample_poly to 100 Hz
         │
         ▼  build_lag_matrix  [lags: −50 ms … +500 ms]
         │
         ▼  RidgeCV  (alpha grid, inner 5-fold CV for λ selection)
         │
         ▼  K-fold outer CV  (k=5, contiguous folds)
         │
    R², Pearson r, TRF weights  per channel
```

**Temporal lags** (default −50 ms … +500 ms at 100 Hz = 56 lags)  
× 2 features = **112 regressors** per time point.

Negative lags let the model capture any anticipatory MEG signal; positive lags
capture the neural response up to 500 ms after the visual input.

---

## Output files

| File | Description |
|------|-------------|
| `sub001_encoding_results.npz` | R², r, weights, fold_r2, lags, labels |
| `sub001_r2_summary.csv`       | Per-channel R² sorted descending |
| `run_config.json`             | All CLI arguments (reproducibility) |
| `*_features.png`              | Feature timeseries per run (sanity check) |
| `*_r2_dist.png`               | Sorted R² bar + histogram |
| `*_trf.png`                   | TRF weights for top-10 channels |
| `*_pred_vs_actual.png`        | Predicted vs actual MEG overlay |
| `*_fold_reliability.png`      | Per-fold R² box plot |
| `*_feature_meg_corr.png`      | Zero-lag feature–MEG correlation |

---

## Sanity checks

A "healthy" preprocessing will show:

1. **Feature timeseries** (`*_features.png`): RMS varies smoothly; SSIM dips at
   scene cuts. Both should look like sensible low-frequency envelopes.

2. **R² distribution** (`*_r2_dist.png`): Most channels near 0, a subset of
   visual/occipital sensors clearly above 0 (typically R² 0.01–0.10 is realistic
   for simple luminance models). If *all* channels are near 0 → model issue or
   data problem. If *all* are high → possible data leakage.

3. **TRF weights** (`*_trf.png`): Should peak around 80–200 ms post-stimulus
   for visual channels. Flat / noisy TRFs in the best channels suggest the
   model is not capturing structure.

4. **Fold reliability** (`*_fold_reliability.png`): All folds should have
   similar distributions. A single outlier fold suggests a trial/run artifact.

5. **Feature–MEG correlation** (`*_feature_meg_corr.png`): Small but non-zero
   positive correlation of RMS with occipital sensors at lag-0 is a good sign
   that the data is reasonable.

---

## Extending the analysis

- **Add optical flow**: replace `extract_visual_features()` to also compute
  `cv2.calcOpticalFlowFarneback()` per frame and append its mean magnitude.
- **More conditions**: `--conditions 1 2 3 4`
- **All subjects**: `sbatch --array=1-72 run_encoding_analysis.sh`
- **Topomaps**: install `mne` and add a call to `mne.viz.plot_topomap(r2, info)`
  after loading the channel positions from the .mat file.
