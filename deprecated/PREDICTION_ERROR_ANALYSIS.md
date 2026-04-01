# V-JEPA Prediction Error Analysis: Why Errors Aren't Monotonic

## The Problem You Observed

Your results showed:
- Pred 4 steps (0.67s): L2 = 55.543 ± ?
- Pred 8 steps (1.33s): L2 = 55.177 ± ?
- Pred 12 steps (2.00s): L2 = 55.000 ± ?

**Expected**: Error should increase monotonically as you predict further into the future.
**Observed**: Error actually *decreased* slightly as you predict further.

This is counterintuitive and suggests a methodological issue, not necessarily a model problem.

---

## Root Cause #1: Ground Truth Encoding Bias ⚠️ CRITICAL

### The Bug

Your original code:
```python
# Encode ALL future frames together
all_future_lat = get_vjepa_latents(model, processor, all_future_frames_raw)

# Then slice for different horizons
for predict_steps in [4, 8, 12]:
    g_lat = all_future_lat[:predict_steps]
```

### Why This Is Wrong

The V-JEPA encoder includes positional embeddings and intra-sequence context modeling. When you encode frames 1-12 together, frame 1's latent representation is influenced by the presence of frames 2-12.

But when you predict 4 steps:
- Context: frames -10 to 0 (encoded together)
- Prediction target: frames 1-4

The ground truth latents for frames 1-4 were encoded as part of a **12-frame sequence**, but the predictions are made from a **10-frame context**. The encoding contexts don't match!

### The Fix

Encode ground truth separately for each prediction horizon:

```python
for predict_steps in PREDICT_TOKENS_LIST:
    # Encode ONLY the frames needed for THIS horizon
    gt_frames = all_future_frames_raw[:predict_steps]
    g_lat = get_vjepa_latents(model, processor, gt_frames)
    
    # Now both context and target are encoded independently
    p_lat = predict_future_latents(model, c_lat, predict_steps)
    
    # Compare apples-to-apples
    cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
```

**This is the most likely source of your non-monotonic behavior.**

---

## Root Cause #2: Averaging Methodology

### The Issue

When you average L2 distances across timesteps:
```python
mean_error = np.mean(l2_dist)  # l2_dist has shape [T]
```

For different T values (4, 8, 12), you're averaging over different numbers of frames. If errors are noisy or heteroscedastic, this can bias results.

### Example of How This Biases Results

Imagine errors at each frame are drawn from slightly different distributions:

```
Pred 4:  [60, 55, 54, 52]  → mean = 55.25
Pred 8:  [60, 55, 54, 52, 51, 50, 49, 48]  → mean = 52.4
Pred 12: [60, 55, 54, 52, 51, 50, 49, 48, 47, 46, 45, 44]  → mean = 50.2
```

If the latent space naturally "converges" to a central point (regression to the mean), longer predictions will show artificially lower average errors.

---

## Best Practices from V-JEPA Literature

The original V-JEPA papers (and violation of expectation work) typically use one of:

### 1. **Maximum Error** (Recommended for Surprise/Expectation Violation)

```python
max_error = np.max(l2_dist)  # Captures peak surprise/violation
```

**Why**: Useful for detecting when a prediction is "most violated." If an unexpected event happens at frame 8, the max error will spike.

### 2. **Error at Specific Frames**

```python
# Final frame only (most distant prediction)
final_frame_error = l2_dist[-1]

# Per-frame analysis
for t in range(len(l2_dist)):
    per_frame_errors[t].append(l2_dist[t])
```

**Why**: Separates "immediate predictability" from "long-term predictability."

### 3. **Error Trajectory Analysis**

```python
# Show how error evolves within each prediction window
# Plot shows: frame 1, frame 2, frame 3, ... frame 12
# across different prediction horizons
```

**Why**: Reveals whether error is front-loaded (hard to predict immediately) or back-loaded (becomes harder as you predict further).

---

## What We Fixed in `vjepa2_sliding_window_v2.py`

### 1. Per-Horizon Ground Truth Encoding ✅
```python
for predict_steps in PREDICT_TOKENS_LIST:
    # Encode ONLY the required frames for this horizon
    gt_frames_for_horizon = all_future_frames_raw[:predict_steps]
    g_lat = get_vjepa_latents(model, processor, gt_frames_for_horizon)
    
    p_lat = predict_future_latents(model, c_lat, predict_steps)
    cos_dist, l2_dist = calculate_metrics(p_lat, g_lat)
```

### 2. Added Maximum Error Tracking ✅
```python
vis_stats[f'pred_{predict_steps}_l2_max'] = np.max(l2_dist)
vis_stats[f'pred_{predict_steps}_cos_max'] = np.max(cos_dist)
```

### 3. Added Per-Frame Error Tracking ✅
```python
for t in range(len(l2_dist)):
    vis_stats[f'pred_{predict_steps}_l2_frame_{t}'] = l2_dist[t]
```

### 4. Enhanced Output Reporting ✅
```
Prediction 4 steps:
  L2 Distance (mean): 55.5430 ± 0.0123
  L2 Distance (max):  62.1234 ± 0.5678
  Per-frame L2 distances:
    Frame 0: 58.2341 ± 0.0456
    Frame 1: 56.1234 ± 0.0789
    Frame 2: 54.8901 ± 0.0654
    Frame 3: 53.2101 ± 0.1234
```

---

## How to Use the Companion Analysis Script

After running the fixed script, analyze results:

```bash
python analyze_prediction_errors.py \
    vjepa_results_sliding/foetsie/frame_metrics_forward.csv \
    --output_dir ./analysis_plots
```

This creates:
1. **error_analysis.png**: Shows mean vs max error across horizons
2. **error_trajectory.png**: Shows per-frame error evolution
3. **correlations.png**: Visual stats vs prediction error scatter plots

---

## What to Expect After the Fix

### Good Signs (Monotonic Degradation):
```
Pred 4 steps:  L2_mean = 52.0, L2_max = 65.0
Pred 8 steps:  L2_mean = 54.0, L2_max = 68.0
Pred 12 steps: L2_mean = 56.0, L2_max = 71.0
```

✅ Errors increase with distance—model predicts farther into uncertain future

### Interesting Patterns (Non-Monotonic But Interpretable):
```
Prediction error drops at frame 8 across all horizons
→ Suggests scenes 0-8 have predictable motion, but frame 8 onwards changes
```

### Warning Signs (Completely Inverted):
```
Pred 4 >> Pred 12
→ Still suggests encoding bias or comparison methodology issue
```

---

## Testing the Fix

To validate the fix works:

1. **Run with forward order**: Baseline
```bash
python vjepa2_sliding_window_v2.py --video foetsie --order forward
```

2. **Run with backward order**: Reversing should show different results
```bash
python vjepa2_sliding_window_v2.py --video foetsie --order backward
```

If encoding bias was the main issue:
- Forward and backward will have different error curves (as expected)
- But both should show monotonic (or interpretable non-monotonic) error growth

---

## The Violation of Expectation Connection

In violation of expectation paradigms (testing if models understand physics):

1. **Plausible scenario**: Model can predict → low surprise (low max error)
2. **Implausible scenario**: Model cannot predict → high surprise (high max error)

The key metric is **maximum error within the prediction window**, because:
- A single violated expectation (one impossible frame) creates a spike
- Average error would dilute this spike across many frames

Your updated script now captures both mean (general difficulty) and max (peak surprise), allowing you to run these kinds of analyses.

---

## References

- Original V-JEPA: [facebook/vjepa](https://github.com/facebookresearch/vjepa)
- Violation of Expectation paradigm: Baillargeon et al. (infant cognition work)
- Physical reasoning with deep learning: Fragkiadaki et al., Mottaghi et al.

