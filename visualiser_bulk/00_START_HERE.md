# V-JEPA2 Analysis Suite - START HERE 🚀

## What You Have

A **complete, publication-ready analysis pipeline** for understanding the relationships between visual content properties and V-JEPA2 video prediction difficulty.

### Three Components:

1. **`analyze_vjepa2_results.py`** – Batch visualization engine
   - Generates 7 plots + statistics per video automatically
   - One command: `python analyze_vjepa2_results.py`
   - Output: `analysis_figures/` with organized results

2. **`interactive_analysis.py`** – Manual exploration tool
   - Menu-driven interface to ask specific questions
   - 5 analysis modes: compare, window, find, correlate
   - One command: `python interactive_analysis.py`

3. **Documentation**
   - `ANALYSIS_TOOLS_SUMMARY.md` – Full overview (this structure)
   - `ANALYSIS_GUIDE.md` – Detailed plot explanations
   - `QUICK_REFERENCE.md` – One-page cheat sheet

---

## 30-Second Quick Start

```bash
# 1. Generate all plots (5-15 minutes depending on video count)
python analyze_vjepa2_results.py

# 2. Open analysis_figures/ and browse the plots

# 3. For specific questions, use interactive tool
python interactive_analysis.py
```

That's it! You now have:
- Correlation heatmaps
- Time series plots
- Windowed analysis
- Scatter relationships
- Distributions
- Complexity analysis
- Cross-video comparison

---

## The 7 Automatic Plots

Each video gets these visualizations automatically:

| # | Plot | Shows | Answers |
|---|------|-------|---------|
| **01** | Correlation Matrix | All metric relationships | Which metrics drive predictability? |
| **02** | Full Time Series | Metrics across entire video | When do prediction failures occur? |
| **03** | Windowed Analysis | Three 100-frame close-ups | What patterns exist locally? |
| **04** | Scatter Relationships | 6 metric pairs | How do visual features affect predictions? |
| **05** | Rolling Correlation | Motion ↔ Unpredictability over time | Does motion relationship stay stable? |
| **06** | Distributions | Histograms of all metrics | Is my video homogeneous or diverse? |
| **07** | Complexity Analysis | Synthetic complexity vs predictability | Do complex scenes resist prediction? |
| **+** | Cross-Video Summary | All videos compared | Which videos are hardest to predict? |

---

## 5 Interactive Analysis Modes

Run `python interactive_analysis.py` to access:

1. **Quick Compare** – Plot any two metrics against each other
   - Time series overlay + scatter plot + correlation
   - Perfect for: "Does optical flow correlate with unpredictability?"

2. **Extract Window** – Zoom into specific frame ranges
   - 4-panel detail view of your selected frames
   - Perfect for: "Show me frames 500-600 in detail"

3. **Find Interesting Frames** – Automatically find anomalies
   - Lists frame indices of most/least predictable moments
   - Perfect for: "Which frames does the model struggle with?"

4. **Correlation Analysis** – See all correlations with one metric
   - Bar chart ranked by correlation strength
   - Perfect for: "What drives 8-step unpredictability?"

5. **Cross-Video Comparison** – Already in batch output
   - See which videos are hardest/easiest to predict

---

## Key Metrics You'll See

### Input: Visual Statistics (computed per frame window)
- **`ctx_rms_contrast`** (0-1): Texture complexity
- **`ctx_brightness`** (0-1): Mean pixel brightness  
- **`ctx_edge_content`** (0-1): Edge density
- **`ctx_optical_flow_magnitude`** (0-∞): Motion magnitude

### Output: Predictability Metrics
- **`pred_4_cos_mean`** (0-1): Unpredictability 4 steps ahead (↓ = easier)
- **`pred_8_cos_mean`** (0-1): Unpredictability 8 steps ahead
- **`pred_12_cos_mean`** (0-1): Unpredictability 12 steps ahead
- (Plus `_std` and `_l2` variants)

### Interpretation
```
Lower pred_*_cos_mean = EASIER to predict
Higher pred_*_cos_mean = HARDER to predict

Example:
  pred_8_cos_mean = 0.3 → Very predictable
  pred_8_cos_mean = 0.7 → Very unpredictable
```

---

## Expected Questions & Answers

### "How do I get started?"
→ Run: `python analyze_vjepa2_results.py`
→ Then: Open `analysis_figures/` and browse

### "Which frames does the model struggle with?"
→ Run: `python interactive_analysis.py`
→ Command 4: Find interesting frames (top 10 hardest)

### "Does motion cause unpredictability?"
→ Check: Plot 05 (rolling_correlation.png)
→ If mostly positive: YES, motion makes prediction harder

### "What visual features matter most?"
→ Run: `python interactive_analysis.py`
→ Command 5: Correlation analysis with `pred_8_cos_mean`

### "Can I zoom into a specific region?"
→ Run: `python interactive_analysis.py`
→ Command 3: Extract window (specify frame range)

### "How do my videos compare?"
→ Check: `analysis_figures/_summary/cross_video_comparison.png`

### "I want publication-quality figures"
→ Edit: `analyze_vjepa2_results.py` line ~30
→ Change: `dpi=150` to `dpi=300`
→ Re-run: `python analyze_vjepa2_results.py`

---

## Complete Workflow Example

### Scenario: "Find out why some video moments are hard to predict"

**Step 1: Overview**
```bash
python analyze_vjepa2_results.py
# Wait 5-15 minutes
```

**Step 2: Identify problem video**
```
Open: analysis_figures/_summary/cross_video_comparison.png
Look: Which video has highest average pred_8_cos_mean?
→ Let's say it's "geiser"
```

**Step 3: Understand drivers**
```bash
python interactive_analysis.py
Command: 5 (Correlation analysis)
Video: geiser
Target: pred_8_cos_mean
→ See bar chart of what correlates with unpredictability
```

**Step 4: Inspect relationships**
```bash
python interactive_analysis.py
Command: 2 (Quick compare)
Video: geiser
Metric 1: ctx_optical_flow_magnitude
Metric 2: pred_8_cos_mean
→ See time series + scatter + correlation coefficient
```

**Step 5: Find problem frames**
```bash
python interactive_analysis.py
Command: 4 (Find interesting frames)
Video: geiser
Metric: pred_8_cos_mean
Threshold: top
N: 10
→ Get list of 10 hardest-to-predict frame indices
```

**Step 6: Zoom in**
```bash
python interactive_analysis.py
Command: 3 (Extract window)
Video: geiser
Start: 523 (from step 5)
End: 623
→ See detailed 100-frame window around problem area
```

**Step 7: Analyze**
```python
# Or do custom Python analysis:
import pandas as pd
df = pd.read_csv('vjepa_results_sliding/geiser/frame_metrics.csv')

# What's different about frames 520-630?
region = df[(df['frame_index'] >= 520) & (df['frame_index'] <= 630)]
print("Average optical flow:", region['ctx_optical_flow_magnitude'].mean())
print("Average unpredictability:", region['pred_8_cos_mean'].mean())
```

---

## Output File Structure

```
analysis_figures/
├── _summary/
│   ├── cross_video_comparison.png      ← Start here for overview
│   └── video_statistics_summary.csv    ← Numerical summary
│
├── deurinhuis/
│   ├── 01_correlation_matrix.png       ← Metric relationships
│   ├── 02_timeseries_full.png          ← Full video metrics
│   ├── 03_windowed_analysis.png        ← Zoomed segments
│   ├── 04_scatter_relationships.png    ← Pairwise plots
│   ├── 05_rolling_correlation.png      ← Motion effect over time
│   ├── 06_distributions.png            ← Histograms
│   ├── 07_complexity_analysis.png      ← Complexity vs pred
│   └── statistics.txt                  ← Text summary
│
├── trap/
│   └── [same 8 files...]
│
└── [other videos...]
```

---

## Common Insights to Look For

### Insight 1: Motion Drives Unpredictability
**Evidence:** Plot 05 (rolling_correlation) mostly positive
**Meaning:** Faster scenes are harder to predict
**Action:** Consider motion-aware model improvements

### Insight 2: Distinct Video Segments
**Evidence:** Plot 06 (distributions) bimodal/skewed
**Meaning:** Video has distinct "static" and "action" parts
**Action:** Analyze separately for better insights

### Insight 3: Complexity Doesn't Explain Everything
**Evidence:** Plot 04 (scatter) show weak correlations with contrast
**Meaning:** Texture/contrast alone doesn't affect predictability
**Action:** Look deeper into motion and appearance changes

### Insight 4: Longer Predictions Are Harder
**Evidence:** pred_12_cos_mean > pred_8_cos_mean > pred_4_cos_mean
**Meaning:** Longer extrapolations accumulate uncertainty (expected!)
**Action:** Report all three horizons in your work

### Insight 5: Scene Transitions Cause Unpredictability
**Evidence:** Rolling correlation plot shows sharp changes
**Meaning:** Relationship between motion and unpredictability varies
**Action:** Identify transition points for segmentation

---

## Tips for Effective Analysis

### Tip 1: Start Broad
```
1. Run batch analysis (2D overview of all videos)
2. Check cross-video summary
3. Pick interesting video
4. Then zoom in with interactive tool
```

### Tip 2: Look for Patterns, Not Noise
```
- Ignore isolated spikes; look for sustained changes
- Compare across prediction horizons (4, 8, 12)
- Check if patterns repeat across multiple videos
```

### Tip 3: Combine Visual & Quantitative
```
- Use plots to identify interesting regions
- Use correlations to quantify relationships
- Load CSV for custom calculations
```

### Tip 4: Document Findings
```
- Screenshot interesting plots with annotations
- Record frame indices of important events
- Save correlation values for reporting
```

### Tip 5: Validate Interpretations
```
- If motion seems important, check multiple videos
- If relationship seems inconsistent, investigate why
- Use interactive tool to drill down on anomalies
```

---

## Integration with Your Research

### Use Case 1: Model Evaluation
```
Compare different model versions:
→ Run each with sliding window analysis
→ Compare cross-video summaries
→ Which model handles motion better?
```

### Use Case 2: Dataset Characterization
```
Understand your video dataset:
→ Generate batch analysis for all videos
→ Look at distribution histograms
→ Identify easy/hard content types
```

### Use Case 3: Failure Analysis
```
Debug when model struggles:
→ Find interesting frames (command 4)
→ Zoom into those regions (command 3)
→ Understand why predictions fail
```

### Use Case 4: Publication Figures
```
Create paper-quality visualizations:
→ Run batch analysis at dpi=300
→ Select most informative plots
→ Export as PDF/PNG for publication
```

### Use Case 5: Custom Analysis
```
Ask specific research questions:
→ Load CSVs with pandas
→ Filter by metric values
→ Create custom visualizations
```

---

## Command Reference

### Batch Analysis
```bash
python analyze_vjepa2_results.py
# Generates all plots for all videos
# Output: analysis_figures/ with 8 plots per video
```

### Interactive Analysis
```bash
python interactive_analysis.py
# Menu interface with 5 modes
# Commands 1-5, 0 to exit
```

### Load Results in Python
```python
import pandas as pd

# Load one video
df = pd.read_csv('vjepa_results_sliding/deurinhuis/frame_metrics.csv')

# Find hardest frames
hard = df.nlargest(10, 'pred_8_cos_mean')
print(hard[['frame_index', 'pred_8_cos_mean', 'ctx_optical_flow_magnitude']])

# Load all videos
from pathlib import Path
dfs = [pd.read_csv(p) for p in Path('vjepa_results_sliding').glob('*/frame_metrics.csv')]
combined = pd.concat(dfs)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No videos found" | Check `vjepa_results_sliding/` directory exists with CSVs |
| Plots empty/missing | Check for NaN values in data: `df.isnull().sum()` |
| Memory error | Reduce video count or analyze one-by-one |
| Slow execution | For very long videos, skip optional plots or subset frames |
| File not found | Check paths are relative to working directory |

---

## Files You Have

```
analyze_vjepa2_results.py      ← Batch plot generator (RUN THIS FIRST)
interactive_analysis.py         ← Interactive explorer
ANALYSIS_TOOLS_SUMMARY.md       ← Full technical overview
ANALYSIS_GUIDE.md               ← Detailed plot explanations
QUICK_REFERENCE.md              ← One-page cheat sheet
00_START_HERE.md                ← This file!
```

---

## Next Actions

### Right Now
1. ✅ Copy scripts to your computer/HPC
2. ✅ Read this file (you're doing it!)

### In 5 Minutes
3. Run: `python analyze_vjepa2_results.py`
4. Browse: `analysis_figures/` folder

### In 30 Minutes
5. Open: `analysis_figures/_summary/cross_video_comparison.png`
6. Identify: Which video is most interesting?

### In 1 Hour
7. Run: `python interactive_analysis.py`
8. Explore: Use commands 1-5 to drill down

### For Deep Analysis
9. Load CSVs in Python for custom analysis
10. Combine findings across multiple videos

---

## Key Takeaways

✅ **Two complementary tools:**
- Batch analysis for automatic visualization
- Interactive tool for manual exploration

✅ **Seven publication-quality plots per video**
- Correlations, time series, scatter plots, distributions
- Cross-video comparisons

✅ **Understand what drives predictability**
- Visual statistics vs. model unpredictability
- Temporal patterns and scene-specific effects

✅ **Find problem frames and regions**
- Automatically identify hardest-to-predict moments
- Zoom in for detailed analysis

✅ **Publication-ready outputs**
- High-quality figures for papers/presentations
- Exportable statistics and correlations

---

## You're Ready!

```
python analyze_vjepa2_results.py
```

Then explore and enjoy! 🎬📊🔍

---

**Questions?** See:
- `QUICK_REFERENCE.md` for cheat sheet
- `ANALYSIS_GUIDE.md` for detailed explanations
- Comments in the source code for implementation details

**Last updated:** March 2026
**For:** V-JEPA2 sliding window analysis results
**Requires:** pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

Happy analyzing! 🚀
