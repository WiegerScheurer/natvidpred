# V-JEPA2 Analysis & Visualization Guide

## Overview

You now have **two powerful tools** for analyzing the V-JEPA2 sliding window results:

1. **`analyze_vjepa2_results.py`** – Batch analysis (generates all plots automatically)
2. **`interactive_analysis.py`** – Interactive exploration (manually select metrics and videos)

---

## Quick Start

### Automatic Analysis (Recommended First)

Generate all visualizations for all videos in one command:

```bash
python analyze_vjepa2_results.py
```

**Output structure:**
```
analysis_figures/
├── _summary/
│   ├── cross_video_comparison.png
│   └── video_statistics_summary.csv
├── deurinhuis/
│   ├── 01_correlation_matrix.png
│   ├── 02_timeseries_full.png
│   ├── 03_windowed_analysis.png
│   ├── 04_scatter_relationships.png
│   ├── 05_rolling_correlation.png
│   ├── 06_distributions.png
│   ├── 07_complexity_analysis.png
│   └── statistics.txt
├── trap/
│   └── [same structure...]
└── [other videos...]
```

### Interactive Analysis

For detailed exploration of specific metrics:

```bash
python interactive_analysis.py
```

**Interactive menu:**
```
1. List videos
2. Quick compare (two metrics)
3. Extract window (specific frame range)
4. Find interesting frames
5. Correlation analysis
0. Exit
```

---

## Visualization Guide

### 1. Correlation Matrix (`01_correlation_matrix.png`)

**What it shows:** How all visual statistics and predictability metrics relate to each other

**Interpretation:**
- **Red/Warm colors**: Positive correlation (metrics increase together)
- **Blue/Cool colors**: Negative correlation (one increases while other decreases)
- **Intensity**: Strength of relationship (darker = stronger)

**Example insights:**
- High positive correlation between brightness and RMS contrast → scene gets brighter AND textured
- Negative correlation between optical flow and 4-step cosine distance → faster motion is MORE predictable (counter-intuitive but common in videos)
- Correlation increases with prediction horizon (4-step < 8-step < 12-step) → longer-term predictions are less predictable

---

### 2. Full Time Series (`02_timeseries_full.png`)

**What it shows:** All metrics across the entire video as line plots

**Four subplots:**
1. **Visual Statistics**: RMS contrast, brightness, edge content (stacked)
2. **Optical Flow**: Motion magnitude (shaded area chart)
3. **Cosine Distance**: Unpredictability at different horizons (4, 8, 12 steps)
4. **L2 Distance**: Latent space distance (alternative metric)

**Interpretation:**
- **Spikes in optical flow** → scene cuts, sudden camera movement, or high action
- **Peaks in cosine distance** → moments where the model struggles to predict future
- **Correlated peaks** → predictability drops during high-motion scenes (expected)
- **Anti-correlated patterns** → complex scenes that look simple by motion metrics

---

### 3. Windowed Analysis (`03_windowed_analysis.png`)

**What it shows:** Three 100-frame windows at different points in the video (start, middle, end)

**Three columns per window:**
1. **Visual stats** over time (local detail)
2. **Motion vs Unpredictability** (dual-axis to see coupling)
3. **Prediction horizons** (4, 8, 12-step unpredictability)

**Interpretation:**
- See **fine-grained temporal patterns** in shorter windows
- Identify **local anomalies** (e.g., one frame's spike in unpredictability)
- Compare **temporal dynamics** across different video segments
- Spot **stable vs. chaotic** regions

---

### 4. Scatter Relationships (`04_scatter_relationships.png`)

**What it shows:** 6 pairwise relationships as scatter plots with trend lines

**Plots:**
1. Motion vs Unpredictability (8-step)
2. Contrast vs Unpredictability (8-step)
3. Brightness vs Unpredictability (8-step)
4. Edge Content vs Unpredictability (8-step)
5. Motion vs Unpredictability (4-step)
6. Contrast vs Unpredictability (4-step)

**Interpretation:**
- **Tight cluster along trend line** → strong linear relationship
- **Scattered cloud** → non-linear or weak relationship
- **R value**: Pearson correlation (-1 to +1)
  - Close to 0 = no relationship
  - Close to ±1 = strong relationship
- **Color gradient** (frame index) shows if relationships change over time

---

### 5. Rolling Correlation (`05_rolling_correlation.png`)

**What it shows:** How the relationship between optical flow and unpredictability changes over time

**Interpretation:**
- **Peaks at +1** → motion strongly predicts unpredictability
- **Valleys at -1** → motion inversely predicts unpredictability (rare)
- **Crosses zero** → relationship changes direction
- **Smooth vs jagged** → stable vs. unstable relationship
- **Per-horizon lines** show whether effect strengthens with longer predictions

**Insight:** If this line is mostly positive, it means faster motion → harder to predict. If negative, scenes move erratically relative to predictability.

---

### 6. Distributions (`06_distributions.png`)

**What it shows:** Histograms of all metrics (top: visual stats, bottom: unpredictability)

**Interpretation:**
- **Normal distribution** → metric varies smoothly throughout video
- **Bimodal/multi-modal** → video has distinct "modes" (e.g., static + action scenes)
- **Skewed right** → most frames low-value with some high spikes
- **Skewed left** → most frames high-value with occasional low dips
- **μ (mean) and σ (std dev)** tell you the average and variability

**Example:**
- Highly skewed optical flow histogram → video is mostly static with brief action
- Uniform unpredictability distribution → consistently hard to predict throughout

---

### 7. Complexity Analysis (`07_complexity_analysis.png`)

**What it shows:** Synthetic "complexity score" (normalized combination of all visual metrics) vs. predictability

**Two subplots:**
1. **Time series**: Complexity (shaded) overlaid with unpredictability (line)
2. **Scatter plot**: Direct relationship with trend line

**Interpretation:**
- **Positive correlation** → visually complex scenes are harder to predict (expected)
- **Negative correlation** → counter-intuitive; may indicate motion-driven unpredictability
- **Time series patterns** show whether complexity "leads" or "lags" unpredictability

---

### 8. Cross-Video Comparison (`_summary/cross_video_comparison.png`)

**What it shows:** Bar charts comparing all videos side-by-side

**Two plots:**
1. **Visual Statistics**: Which videos are most/least complex on average
2. **Predictability Metrics**: Which videos are easier/harder to predict

**Use case:**
- Rank videos by difficulty
- Identify dataset characteristics
- Compare model performance across different content types

---

## Interactive Tool Usage

### Command 2: Quick Compare

Compare any two metrics (e.g., optical flow vs. 8-step unpredictability):

```bash
python interactive_analysis.py
# → Select command 2
# → Choose video
# → Choose two metrics from the list
```

**Output:** Two-panel figure with time series and scatter plot + correlation

### Command 3: Extract Window

Zoom into a specific frame range (e.g., frames 500-600):

```bash
python interactive_analysis.py
# → Select command 3
# → Choose video
# → Enter start frame: 500
# → Enter end frame: 600
```

**Output:** Four-panel detailed view of that segment

**Tip:** Use this when you see interesting patterns in the full time series!

### Command 4: Find Interesting Frames

Automatically find the most/least predictable frames:

```bash
python interactive_analysis.py
# → Select command 4
# → Choose video
# → Choose metric (e.g., pred_8_cos_mean)
# → Select 'top' for unpredictable or 'bottom' for predictable
# → Enter number of frames (e.g., 10)
```

**Output:** Table of frame indices + visualization with outliers highlighted

**Use case:** 
- "What frames does the model struggle with most?"
- "Find the easiest-to-predict moments"

### Command 5: Correlation Analysis

See how every metric correlates with one target metric:

```bash
python interactive_analysis.py
# → Select command 5
# → Choose video
# → Choose target metric (e.g., pred_8_cos_mean)
```

**Output:** Horizontal bar chart ranked by correlation strength

**Interpretation:**
- Metrics on the right → positive correlation
- Metrics on the left → negative correlation
- `**` → highly significant (p < 0.001)
- `*` → moderately significant (p < 0.05)

---

## Interpreting Key Findings

### Scenario 1: Strong Positive Correlation (Optical Flow → Unpredictability)

**Meaning:** Faster-moving scenes are harder to predict

**Why:** Motion introduces variability; the model needs to extrapolate motion patterns

**Action:** 
- Consider motion-aware loss terms in model training
- Separate analysis for static vs. dynamic scenes

### Scenario 2: Weak/Negative Correlation (Contrast → Unpredictability)

**Meaning:** Texture/contrast doesn't strongly affect predictability

**Why:** V-JEPA focuses on semantic features, not low-level texture

**Action:**
- High-contrast doesn't mean hard-to-predict
- Focus on object motion/appearance changes instead

### Scenario 3: Bimodal Optical Flow Distribution

**Meaning:** Video has clear "static" and "action" segments

**Why:** Different content types (e.g., talking head + scene cuts)

**Action:**
- Analyze static vs. dynamic predictability separately
- Use as segmentation signal for downstream tasks

### Scenario 4: Rolling Correlation Switches Sign

**Meaning:** Relationship between motion and predictability changes over time

**Why:** Different scene types (e.g., camera pan vs. object motion)

**Action:**
- Investigate those time ranges manually
- May indicate scene boundaries or content transitions

---

## CSV Output Format

Each video's results are also exported as:
- **`frame_metrics.csv`**: Raw frame-level metrics (for your own analysis)
- **`statistics.txt`**: Summary statistics and key correlations

### Columns in `frame_metrics.csv`:

```
frame_index                          Frame window starting position
ctx_rms_contrast                     Texture complexity (0-1)
ctx_brightness                       Mean brightness (0-1)
ctx_edge_content                     Edge density (0-1)
ctx_optical_flow_magnitude           Motion magnitude (pixels/frame)
pred_4_cos_mean                      Unpredictability 4 steps ahead (lower = easier)
pred_4_cos_std                       Variance in 4-step predictions
pred_4_l2_mean                       L2 distance 4 steps ahead
pred_4_l2_std                        L2 variance
pred_8_cos_mean                      Unpredictability 8 steps ahead
[... pred_8_cos_std, pred_8_l2_mean, pred_8_l2_std ...]
pred_12_cos_mean                     Unpredictability 12 steps ahead
[... pred_12_cos_std, pred_12_l2_mean, pred_12_l2_std ...]
```

### Load and analyze in Python:

```python
import pandas as pd

# Load results
df = pd.read_csv('vjepa_results_sliding/deurinhuis/frame_metrics.csv')

# Find hardest-to-predict moments
hard = df.nlargest(10, 'pred_8_cos_mean')
print(hard[['frame_index', 'ctx_optical_flow_magnitude', 'pred_8_cos_mean']])

# Compute average unpredictability
print(f"Average 8-step unpredictability: {df['pred_8_cos_mean'].mean():.4f}")

# Group by optical flow levels
df['flow_level'] = pd.cut(df['ctx_optical_flow_magnitude'], bins=5)
print(df.groupby('flow_level')['pred_8_cos_mean'].mean())
```

---

## Advanced Tips

### Custom Analysis

Modify `analyze_vjepa2_results.py` to:

```python
# Analyze only one video
VIDEO_NAMES = ["deurinhuis"]

# Change window size for windowed plots
WINDOW_SIZE = 200

# Disable specific plots by commenting out in generate_report()
# (e.g., skip GIF generation if only need CSVs)
```

### Batch Processing

Generate analysis for subset of videos:

```bash
# Edit analyze_vjepa2_results.py to set VIDEO_NAMES = ["video1", "video2"]
python analyze_vjepa2_results.py
```

### Export Results for Publication

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('vjepa_results_sliding/deurinhuis/frame_metrics.csv')

# High-resolution publication figure
fig = plt.figure(figsize=(10, 6), dpi=300)
# ... create your plot ...
fig.savefig('publication_figure.pdf', bbox_inches='tight')
```

### Combine Multiple Videos

```python
import pandas as pd
from pathlib import Path

results_dir = Path('vjepa_results_sliding')
dfs = []

for video_dir in results_dir.iterdir():
    csv = video_dir / 'frame_metrics.csv'
    if csv.exists():
        df = pd.read_csv(csv)
        df['video'] = video_dir.name
        dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Now analyze across all videos
print(combined.groupby('video')['pred_8_cos_mean'].agg(['mean', 'std']))
```

---

## Troubleshooting

**Q: "No videos found in results directory!"**
- **Solution:** Ensure `vjepa_results_sliding/` exists with subdirectories containing `frame_metrics.csv`

**Q: Memory error when loading many large videos**
- **Solution:** Analyze videos individually in interactive mode instead of all at once

**Q: Plots look strange or missing data**
- **Solution:** Check if CSV has NaN values (`df.isnull().sum()`) and handle in custom analysis

**Q: Want to zoom into one small video region?**
- **Solution:** Use interactive tool command 3 (Extract window) with specific frame range

---

## Publication-Ready Figures

All plots use:
- Professional color schemes (colorblind-friendly)
- Large fonts for readability
- High DPI (150 for screen, edit to 300 for print)
- Grid lines for easy value reading
- Clear legends and axis labels

**To edit figures for publication:**

```python
# Open PNG in matplotlib
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('analysis_figures/deurinhuis/02_timeseries_full.png')
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)
ax.axis('off')
# ... edit ...
```

---

## Next Steps

1. **Run batch analysis:** `python analyze_vjepa2_results.py`
2. **Browse figures** in `analysis_figures/` subdirectories
3. **Explore interactively** for specific questions: `python interactive_analysis.py`
4. **Load CSVs** for custom analysis in your research code
5. **Share findings** with your lab/collaborators

---

## Questions or Custom Visualizations?

The scripts are modular. To add custom plots:

1. Define a `plot_my_custom_plot(df, video_name)` function
2. Call it in `generate_report()` before saving
3. Save output with `plt.savefig()`

Example:

```python
def plot_my_custom_plot(df, video_name):
    fig, ax = plt.subplots()
    # ... your plot code ...
    return fig

# In generate_report():
print("Generating custom plot...")
fig = plot_my_custom_plot(df, video_name)
fig.savefig(video_fig_dir / "08_my_custom_plot.png", dpi=150, bbox_inches='tight')
plt.close(fig)
```

---

**Good luck with your analysis!** 🎬📊

