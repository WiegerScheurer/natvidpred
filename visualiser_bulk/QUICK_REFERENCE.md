# V-JEPA2 Analysis Tools - Quick Reference Card

## Files Overview

| File | Purpose | When to Use |
|------|---------|------------|
| `analyze_vjepa2_results.py` | Generate ALL plots automatically | First-time analysis, full report |
| `interactive_analysis.py` | Manually explore specific metrics | Deep diving, hypothesis testing |
| `analyze_vjepa2_results.py` | Main workhorse | Daily use |

---

## Running the Tools

### Batch Analysis (Generate Everything)
```bash
python analyze_vjepa2_results.py
# Output: analysis_figures/ with all plots
```

### Interactive Exploration
```bash
python interactive_analysis.py
# Menu-driven interface for selective analysis
```

---

## Plot Types & What They Show

### 01: Correlation Matrix
- **Shows:** All metric relationships
- **Look for:** Blocks of color (strong patterns)
- **Interpret:** Red = positive, Blue = negative
- **Action:** Identify key driver metrics

### 02: Full Time Series
- **Shows:** 4 subplots of metrics over entire video
- **Look for:** Peaks, valleys, synchronized spikes
- **Interpret:** When do problems occur?
- **Action:** Identify temporal patterns

### 03: Windowed Analysis (3 windows)
- **Shows:** Detailed 100-frame segments
- **Look for:** Local anomalies
- **Interpret:** Compare start/middle/end
- **Action:** Zoom into interesting regions

### 04: Scatter Relationships
- **Shows:** 6 pairwise metric correlations
- **Look for:** Points along trend line
- **Interpret:** Strength and direction of effects
- **Action:** Find which visual stats matter most

### 05: Rolling Correlation
- **Shows:** Time-varying correlation between motion and unpredictability
- **Look for:** Positive/negative peaks, zero crossings
- **Interpret:** Does motion make frames harder to predict?
- **Action:** Identify scene types with different dynamics

### 06: Distributions
- **Shows:** Histograms of all metrics
- **Look for:** Skewness, modality, outliers
- **Interpret:** Is your video diverse or homogeneous?
- **Action:** Classify video content type

### 07: Complexity Analysis
- **Shows:** Synthetic complexity score vs. predictability
- **Look for:** Overall trend and coupling
- **Interpret:** Do complex scenes resist prediction?
- **Action:** Decide if complexity is a predictor

### 08: Cross-Video Summary
- **Shows:** Side-by-side comparison of all videos
- **Look for:** Ranking, outliers
- **Interpret:** Which videos are hardest/easiest?
- **Action:** Characterize your dataset

---

## Common Questions & Quick Answers

### "Why is unpredictability so high here?"
→ Look at plots 02 & 05. Check optical flow and edge content at that frame.

### "Do faster scenes predict worse?"
→ Check plot 05 (rolling correlation). If mostly positive, yes.

### "Which visual metrics matter most?"
→ Check plot 04 (scatter). Tighter trend line = stronger effect.

### "Is my video homogeneous or diverse?"
→ Check plot 06 (distributions). Skewed/bimodal = diverse.

### "How do my videos compare?"
→ Check cross-video summary (plot 08).

---

## Key Metrics Definitions

| Metric | Range | Meaning | Example |
|--------|-------|---------|---------|
| `ctx_rms_contrast` | 0-1 | Texture complexity | 0.1 = smooth, 0.9 = busy |
| `ctx_brightness` | 0-1 | Mean pixel value | 0.2 = dark, 0.8 = bright |
| `ctx_edge_content` | 0-1 | Edge density | 0.1 = smooth, 0.7 = sharp objects |
| `ctx_optical_flow_magnitude` | 0-∞ | Motion speed | 0.5 = still, 5.0 = fast action |
| `pred_*_cos_mean` | 0-1 | Unpredictability | 0.2 = easy, 0.8 = hard |
| `pred_*_l2_mean` | 0-∞ | Latent distance | Lower = more predictable |

---

## Interpretation Cheat Sheet

### Correlation Matrix
```
Strong positive (+0.8): metrics increase together
Moderate positive (+0.4): weak coupling
Near zero (0.0): independent
Moderate negative (-0.4): inverse relationship
Strong negative (-0.8): inverse relationship
```

### Time Series
```
Rising trend: metrics increasing throughout video
Falling trend: metrics decreasing throughout video
Flat: stable throughout
Spikes: sudden events or scene changes
```

### Scatter Plot
```
Tight cluster → strong relationship
Scattered cloud → weak relationship
Outliers → unusual frames
Trend line slope > 1 → steep relationship
Trend line slope ≈ 0 → no relationship
```

---

## Interactive Commands Quick Ref

```
1 = List videos
2 = Compare two metrics (time series + scatter)
3 = Zoom into frame range
4 = Find most/least predictable frames
5 = Show all correlations with one metric
0 = Exit
```

### Command 2 Example
```bash
→ Select video "trap"
→ Metric 1: ctx_optical_flow_magnitude
→ Metric 2: pred_8_cos_mean
# Output: Why does optical flow correlate with unpredictability?
```

### Command 3 Example
```bash
→ Select video "deurinhuis"
→ Start frame: 500
→ End frame: 600
# Output: Detailed view of frames 500-600
```

### Command 4 Example
```bash
→ Select video "geiser"
→ Metric: pred_12_cos_mean
→ Threshold: top
→ N frames: 15
# Output: Most unpredictable moments in the video
```

---

## Output Files

```
analysis_figures/
├── _summary/
│   ├── cross_video_comparison.png          [All videos compared]
│   └── video_statistics_summary.csv        [Numerical summary]
│
├── {video_name}/
│   ├── 01_correlation_matrix.png           [All metric relationships]
│   ├── 02_timeseries_full.png              [Full video metrics]
│   ├── 03_windowed_analysis.png            [Three 100-frame windows]
│   ├── 04_scatter_relationships.png        [6 pairwise plots]
│   ├── 05_rolling_correlation.png          [Motion ↔ Unpredictability]
│   ├── 06_distributions.png                [Histograms]
│   ├── 07_complexity_analysis.png          [Complexity ↔ Predictability]
│   └── statistics.txt                      [Numerical summary]
```

---

## Workflow Examples

### Scenario 1: "Find hardest moments"
```bash
python interactive_analysis.py
→ Command 4
→ Select video
→ Metric: pred_8_cos_mean
→ Threshold: top
→ N: 10
# Now you have frame indices of hardest frames
# Use Command 3 to zoom into those regions
```

### Scenario 2: "Compare two videos"
```bash
# Run batch first
python analyze_vjepa2_results.py

# Compare plot 07 (complexity analysis) side-by-side
# Check cross-video summary (plot 08)
# Videos with similar complexity should have similar predictability
```

### Scenario 3: "Understand what drives predictability"
```bash
python interactive_analysis.py
→ Command 5
→ Select target metric: pred_8_cos_mean
# See bar chart of all correlations
# Metrics on right = positive correlation (make predictions harder)
# Metrics on left = negative correlation (make predictions easier)
```

### Scenario 4: "Find scene boundaries"
```bash
# Check plot 05 (rolling correlation)
# Look for sudden changes in correlation sign
# These likely mark transitions between scene types
```

---

## Pro Tips

1. **Start with batch analysis** (`python analyze_vjepa2_results.py`) to get overview
2. **Then use interactive** (`python interactive_analysis.py`) for deep dives
3. **Export interesting plots** at high DPI (edit source code, change `dpi=150` → `dpi=300`)
4. **Combine CSVs** for cross-video analysis (see ANALYSIS_GUIDE.md)
5. **Look at multiple videos** to understand patterns vs. anomalies

---

## Customization

### Modify window size for detailed analysis
```python
# In analyze_vjepa2_results.py, line ~30
WINDOW_SIZE = 200  # Change from 100 to 200 for larger windows
```

### Analyze only specific videos
```python
# In analyze_vjepa2_results.py, line ~40
VIDEO_NAMES = ["deurinhuis", "trap"]  # Analyze only these
```

### Change color scheme
```python
# In any script, change
plt.style.use('seaborn-v0_8-darkgrid')
# to
plt.style.use('ggplot')  # or 'bmh', 'fivethirtyeight', etc.
```

---

## Python Usage Examples

```python
import pandas as pd

# Load one video
df = pd.read_csv('vjepa_results_sliding/deurinhuis/frame_metrics.csv')

# Find frames with high optical flow and low predictability (motion doesn't explain unpredictability)
anomalies = df[(df['ctx_optical_flow_magnitude'] < 0.5) & 
               (df['pred_8_cos_mean'] > 0.6)]

# Average unpredictability by optical flow level
df['flow_bin'] = pd.cut(df['ctx_optical_flow_magnitude'], bins=5)
print(df.groupby('flow_bin')['pred_8_cos_mean'].mean())

# Load all videos
from pathlib import Path
dfs = [pd.read_csv(p) for p in Path('vjepa_results_sliding').glob('*/frame_metrics.csv')]
combined = pd.concat(dfs)
print(combined['pred_8_cos_mean'].describe())
```

---

## When to Use Each Plot

| Question | Use Plot |
|----------|----------|
| Which metrics correlate? | 01 (Correlation) |
| When do problems occur? | 02 (Time series) |
| Show me a local region | 03 (Windowed) |
| What drives predictability? | 04 (Scatter) |
| Does motion cause unpredictability? | 05 (Rolling corr) |
| How diverse is my video? | 06 (Distributions) |
| Do complex scenes resist prediction? | 07 (Complexity) |
| Which video is hardest? | 08 (Cross-video) |

---

## Metric Correlations to Expect

| Pair | Expected | Why |
|------|----------|-----|
| Optical Flow ↔ Unpredictability | Positive | Motion is inherently harder to predict |
| Contrast ↔ Unpredictability | Weak/None | V-JEPA focuses on semantics, not texture |
| Brightness ↔ Unpredictability | Weak/None | Lighting doesn't strongly affect predictions |
| Edge Content ↔ Unpredictability | Positive | More structure = more stuff to predict |
| 4-step ↔ 8-step Unpredictability | Very Positive | Both measure same thing, different horizons |

---

## Summary

**Three-step process:**
1. Run `python analyze_vjepa2_results.py` → Get all plots
2. Browse `analysis_figures/` → Understand your data
3. Use `python interactive_analysis.py` → Answer specific questions

**Key insight:** Plots show RELATIONSHIPS between visual properties and model predictability.

---

*Last updated: March 2026*
