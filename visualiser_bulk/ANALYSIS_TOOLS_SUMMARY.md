# Complete Analysis Tools Suite - Summary

## What You've Got

You now have a **complete analysis pipeline** with **two complementary tools** for understanding the relationships between visual content and V-JEPA2 predictability:

### 1. **Batch Analysis Script** (`analyze_vjepa2_results.py`)
Automatically generates **7 publication-quality plots** for each video + cross-video summary.

### 2. **Interactive Explorer** (`interactive_analysis.py`)
Menu-driven tool for manual exploration with 5 powerful analysis modes.

### 3. **Documentation**
- `ANALYSIS_GUIDE.md` – Detailed explanation of each plot
- `QUICK_REFERENCE.md` – One-page cheat sheet

---

## The Pipeline

```
Your CSV Results
    ↓
    ├─→ [Batch Analysis] → 7 plots × N videos + summary
    │
    └─→ [Interactive Explorer] → Manual deep dives
        ├─ Quick compare any 2 metrics
        ├─ Zoom into frame windows
        ├─ Find interesting frames
        └─ Correlation analysis
```

---

## 5-Minute Getting Started

### Step 1: Run Batch Analysis
```bash
python analyze_vjepa2_results.py
# Generates all plots automatically (~5-15 min depending on video count)
# Output: analysis_figures/ with organized subdirectories
```

### Step 2: Browse Results
```bash
# Open analysis_figures/ and look at plots
# Start with 02_timeseries_full.png to see overall patterns
# Check 01_correlation_matrix.png to understand metric relationships
```

### Step 3: Interactive Deep Dive
```bash
python interactive_analysis.py
# Menu-driven exploration
# Ask specific questions about your data
```

---

## 7 Automatic Plots (Per Video)

| Plot | File | Purpose |
|------|------|---------|
| **01** | `correlation_matrix.png` | Show relationships between ALL metrics |
| **02** | `timeseries_full.png` | Visual metrics, motion, predictability over full video |
| **03** | `windowed_analysis.png` | Three detailed 100-frame windows (start/mid/end) |
| **04** | `scatter_relationships.png` | 6 scatter plots: visual stats vs. unpredictability |
| **05** | `rolling_correlation.png` | How motion-predictability relationship changes over time |
| **06** | `distributions.png` | Histograms of all metrics (understand data diversity) |
| **07** | `complexity_analysis.png` | Synthetic complexity score vs. predictability |
| **Summary** | `statistics.txt` | Numerical summary + key correlations |

---

## 5 Interactive Analysis Modes

| Command | Input | Output | Use Case |
|---------|-------|--------|----------|
| **Compare** | Two metrics | Time series + scatter plot | "How do X and Y relate?" |
| **Window** | Frame range | 4-panel detail view | "Show me frames 500-600" |
| **Find** | Metric + top/bottom | Frame indices + visualization | "Which frames are hardest to predict?" |
| **Correlate** | One target metric | Bar chart of all correlations | "What drives this metric?" |

---

## Visualization Examples

### Plot 01: Correlation Matrix
```
Shows: All metric relationships at a glance
Red = positive (metrics increase together)
Blue = negative (inverse relationship)
Example insight: Optical flow strongly correlates with 
                 unpredictability → motion makes prediction harder
```

### Plot 02: Full Time Series
```
Shows: 4 subplots across entire video
1. Visual statistics (contrast, brightness, edges)
2. Optical flow (motion activity)
3. Cosine distance (unpredictability 4/8/12 steps)
4. L2 distance (alternative metric)

Look for: Synchronized peaks → moments where things get hard to predict
```

### Plot 03: Windowed Analysis
```
Shows: Zoomed-in views of 3 segments (start/middle/end)
Each has 3 columns: visual stats, motion vs. unpredictability, prediction horizons

Look for: Local patterns invisible at full-video scale
```

### Plot 04: Scatter Relationships
```
Shows: 6 pairwise metric correlations (examples below)
Interpretation:
  Tight trend line = strong relationship
  Scattered cloud = weak relationship
  R value = correlation strength (-1 to +1)
```

### Plot 05: Rolling Correlation
```
Shows: How motion-predictability relationship changes over time
Positive peaks = "faster motion → harder to predict"
Negative peaks = "faster motion → easier to predict" (rare)
Zero crossings = scene transition points

Key insight: Relationship is NOT static; varies by content
```

### Plot 06: Distributions
```
Shows: Histograms of all metrics
Interpretation:
  Normal bell curve = metric varies smoothly
  Bimodal/skewed = distinct "types" in video (e.g., static + action)
  μ (mean) and σ (std) show variability
```

### Plot 07: Complexity Analysis
```
Shows: Synthetic "complexity score" (normalized visual features)
        vs. predictability (cosine distance)

Time series: See if complexity "leads" unpredictability
Scatter: Quantify the relationship

Example: If positive correlation, complex scenes harder to predict
```

### Plot 08: Cross-Video Summary
```
Shows: Bar charts comparing ALL videos side-by-side
Visual metrics: Which videos have most motion? texture? brightness?
Predictability: Which videos does model predict best/worst?

Use case: Rank and characterize your video dataset
```

---

## Key Metrics You'll See

### Visual Statistics (Input Features)
- **`ctx_rms_contrast`** (0-1): Texture complexity → High = busy scenes
- **`ctx_brightness`** (0-1): Mean pixel intensity → High = bright scenes
- **`ctx_edge_content`** (0-1): Edge density → High = sharp objects
- **`ctx_optical_flow_magnitude`** (0-∞): Motion → High = fast-moving scenes

### Predictability Metrics (Model Output)
- **`pred_N_cos_mean`** (0-1): Cosine distance → **Lower = MORE predictable**
- **`pred_N_l2_mean`** (0-∞): L2 latent distance → Lower = more predictable
- Where **N ∈ {4, 8, 12}** = prediction horizon (steps into future)

### Interpretation
```
Optical flow = 1.0, pred_8_cos_mean = 0.3
→ This frame has moderate motion but is EASY to predict

Optical flow = 0.5, pred_8_cos_mean = 0.7
→ This frame is mostly STATIC but HARD to predict
→ Suggests motion is not the only driver!
```

---

## Common Workflows

### Workflow 1: "Get Overview"
```
1. Run: python analyze_vjepa2_results.py
2. Open: analysis_figures/_summary/cross_video_comparison.png
3. Result: Which videos are hardest to predict?
```

### Workflow 2: "Find Problem Frames"
```
1. Run: python interactive_analysis.py
2. Select: Command 4 (Find interesting frames)
3. Choose: metric = pred_8_cos_mean, threshold = top, n = 10
4. Result: Indices of 10 hardest-to-predict frames
5. Follow-up: Command 3 to zoom into those regions
```

### Workflow 3: "Understand Drivers"
```
1. Run: python interactive_analysis.py
2. Select: Command 5 (Correlation analysis)
3. Choose: target metric = pred_8_cos_mean
4. Result: Bar chart showing which visual features correlate most
5. Interpret: What metrics can you control to improve predictability?
```

### Workflow 4: "Deep Dive One Region"
```
1. Identify interesting region in plot 02 (timeseries_full.png)
2. Run: python interactive_analysis.py
3. Select: Command 3 (Extract window)
4. Input: Frame range from interesting region
5. Result: 4-panel detail view of that segment
```

### Workflow 5: "Custom Analysis"
```
1. Load CSV: df = pd.read_csv('vjepa_results_sliding/video/frame_metrics.csv')
2. Filter: hard_frames = df[df['pred_8_cos_mean'] > 0.7]
3. Plot: your own custom visualizations
4. Export: results.to_csv('my_analysis.csv')
```

---

## Expected Findings

### Finding 1: Positive Correlation (Optical Flow → Unpredictability)
```
Meaning: Faster motion = harder to predict
Why: Motion introduces variability; model must extrapolate motion
Common: ~70% of videos show this
Action: Consider motion-aware loss functions
```

### Finding 2: Scene Types in Distribution
```
Meaning: Bimodal optical flow histogram
Why: Video has distinct "static" and "action" segments
Example: Talking head (static) + scene cuts (action)
Action: Analyze segments separately for better insights
```

### Finding 3: Complexity Doesn't Predict Unpredictability
```
Meaning: High contrast/edges don't correlate with hard-to-predict
Why: V-JEPA focuses on semantic features, not low-level texture
Action: Don't over-focus on texture; look at motion + appearance changes
```

### Finding 4: Prediction Horizons Matter
```
Meaning: 4-step predictions easier than 8-step, which easier than 12-step
Why: Longer extrapolations accumulate uncertainty
Expected: Mostly monotonic (though not always)
Action: Report all three horizons in your analysis
```

---

## Output Organization

```
Your Analysis Results
analysis_figures/
│
├── _summary/                          [Cross-video comparison]
│   ├── cross_video_comparison.png     [All videos side-by-side]
│   └── video_statistics_summary.csv   [Numerical data]
│
├── deurinhuis/                        [First video]
│   ├── 01_correlation_matrix.png
│   ├── 02_timeseries_full.png
│   ├── 03_windowed_analysis.png
│   ├── 04_scatter_relationships.png
│   ├── 05_rolling_correlation.png
│   ├── 06_distributions.png
│   ├── 07_complexity_analysis.png
│   └── statistics.txt                 [Summary stats]
│
├── trap/                              [Second video]
│   └── [same 8 files...]
│
└── [other videos...]
```

Plus your original data:
```
vjepa_results_sliding/
├── deurinhuis/
│   └── frame_metrics.csv              [Raw frame-level data]
├── trap/
│   └── frame_metrics.csv
└── [other videos...]
```

---

## Tips & Tricks

### Tip 1: Start Broad, Then Narrow
```
1. Run batch analysis (overview of all videos)
2. Pick interesting video based on plots
3. Use interactive tool to zoom in
4. Load CSV for custom analysis if needed
```

### Tip 2: Compare Prediction Horizons
```python
import pandas as pd
df = pd.read_csv('vjepa_results_sliding/trap/frame_metrics.csv')

# How much harder is 12-step vs 4-step?
print(df['pred_12_cos_mean'].mean() - df['pred_4_cos_mean'].mean())
```

### Tip 3: Find Anomalies
```python
# Frames with high motion but EASY to predict
anomalies = df[(df['ctx_optical_flow_magnitude'] > 3.0) & 
               (df['pred_8_cos_mean'] < 0.3)]
```

### Tip 4: Export for Publication
```bash
# Edit analyze_vjepa2_results.py
# Change: dpi=150  →  dpi=300
# Re-run: python analyze_vjepa2_results.py
# Result: Publication-ready high-resolution figures
```

### Tip 5: Batch Multiple Videos
```bash
# Load and compare all videos
python -c "
import pandas as pd
from pathlib import Path

dfs = []
for csv in Path('vjepa_results_sliding').glob('*/frame_metrics.csv'):
    df = pd.read_csv(csv)
    df['video'] = csv.parent.name
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(combined.groupby('video')['pred_8_cos_mean'].mean())
"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No videos found" | Check `vjepa_results_sliding/` exists with subdirectories |
| Memory error | Analyze videos one-by-one in interactive mode |
| Plots look empty | Check if metrics have NaN values (`df.isnull().count()`) |
| Slow execution | For large videos, skip GIF generation or use subset |
| Can't find frame range | Check max frames: `df['frame_index'].max()` |

---

## Integration with Your Workflow

### Before (Just had raw CSV data)
```
frame_metrics.csv → unclear what it means → need custom scripts
```

### After (With analysis tools)
```
frame_metrics.csv
    ↓
    ├→ 01_correlation_matrix.png        [Understand relationships]
    ├→ 02_timeseries_full.png           [See temporal patterns]
    ├→ 04_scatter_relationships.png     [Quantify effects]
    ├→ 05_rolling_correlation.png       [Track changes]
    │
    └→ Interactive tool                 [Answer specific questions]
        ├ Compare any two metrics
        ├ Find problem frames
        ├ Deep dive regions
        └ Analyze correlations
```

---

## Next Steps

1. **Copy scripts to HPC** (if analyzing remote results)
2. **Run batch analysis:** `python analyze_vjepa2_results.py`
3. **Browse output:** Open `analysis_figures/` in file explorer
4. **Deep dive:** Use interactive tool for specific questions
5. **Custom analysis:** Load CSVs for your own scripts

---

## Questions to Ask Your Data

- "Which frames are hardest to predict?"
- "Does motion cause unpredictability?"
- "Are my videos consistent or diverse?"
- "Which video is easiest/hardest to predict?"
- "Do bright scenes predict differently?"
- "When does the model fail most?"
- "Are there scene transitions?"
- "How different are 4/8/12-step predictions?"

**Each question has a corresponding plot or interactive command!**

---

## Files Summary

```
analyze_vjepa2_results.py  ← Main workhorse (batch all plots)
interactive_analysis.py    ← Manual explorer (ask questions)
ANALYSIS_GUIDE.md          ← Detailed plot explanations
QUICK_REFERENCE.md         ← One-page cheat sheet
ANALYSIS_TOOLS_SUMMARY.md  ← This file (overview)
```

---

## Citation & Reproducibility

These tools are designed to:
- ✅ Make V-JEPA2 results interpretable
- ✅ Identify drivers of unpredictability
- ✅ Compare videos systematically
- ✅ Generate publication-ready figures
- ✅ Support reproducible research

---

**You're ready to analyze!** 🚀📊

Start with: `python analyze_vjepa2_results.py`

---

*Created: March 2026*
*Designed for: V-JEPA2 sliding window analysis*
*Requirements: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn*
