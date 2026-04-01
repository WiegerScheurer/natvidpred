#!/usr/bin/env python3
"""
MEG Encoding Analysis — Visual Feature Temporal Ridge Regression (TRF)
======================================================================
Extracts per-frame visual features (RMS, SSIM) from the stimulus videos,
builds a temporally-lagged (FIR) design matrix, and fits a ridge regression
model to predict MEG sensor responses via k-fold cross-validation.

Condition mapping
-----------------
  1 = attend   forward   → regular video  (part<NN>_24Hz.mp4)
  2 = unattend forward   → regular video
  3 = attend   backward  → reversed video (part<NN>_bw_24Hz.mp4)
  4 = unattend backward  → reversed video

Usage
-----
    python meg_encoding_analysis.py \\
        --subject       1 \\
        --conditions    1 3 \\
        --data_dir      /path/to/mat_files \\
        --video_dir     /path/to/videos \\
        --condition_table /path/to/ConditionTable.csv \\
        --output_dir    /path/to/output \\
        --meg_key       data          \\   # top-level key in the .mat file
        --n_folds       5 \\
        --lag_min      -0.05 \\
        --lag_max       0.50
"""



# ── Imports ────────────────────────────────────────────────────────────────
import os, sys, argparse, logging, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import h5py
import scipy.io as sio
from scipy import signal, stats
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  CLI
# ═══════════════════════════════════════════════════════════════════════════
def build_parser():
    p = argparse.ArgumentParser(
        description='MEG visual encoding analysis (TRF / ridge regression)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--subject',          type=int,   default=1,
                   help='Participant number (matches ConditionTable)')
    p.add_argument('--conditions',       type=int,   nargs='+', default=[1, 3],
                   help='Conditions to include (1=att-fwd, 2=unatt-fwd, '
                        '3=att-bwd, 4=unatt-bwd)')
    p.add_argument('--data_dir',         type=str,   required=True,
                   help='Directory containing sub<NNN>_100Hz_*.mat files')
    p.add_argument('--video_dir',        type=str,   required=True,
                   help='Directory containing ProjectAttention_movie_part*.mp4')
    p.add_argument('--condition_table',  type=str,   required=True,
                   help='Path to ConditionTable.csv')
    p.add_argument('--output_dir',       type=str,   default='./output',
                   help='Where to write results and figures')
    # MAT loading
    p.add_argument('--meg_key',          type=str,   default='data',
                   help='Top-level variable name in the .mat file')
    p.add_argument('--prefer_v73',       action='store_true',
                   help='Prefer the _v7_3.mat file (HDF5) over the plain .mat')
    # Temporal model
    p.add_argument('--meg_fs',           type=float, default=100.0,
                   help='MEG sampling frequency (Hz)')
    p.add_argument('--video_fps',        type=float, default=24.0,
                   help='Video frame rate (fps)')
    p.add_argument('--lag_min',          type=float, default=-0.05,
                   help='Minimum lag for the TRF window (s, can be negative)')
    p.add_argument('--lag_max',          type=float, default=0.50,
                   help='Maximum lag for the TRF window (s)')
    # Ridge / CV
    p.add_argument('--n_folds',          type=int,   default=5,
                   help='Number of cross-validation folds')
    p.add_argument('--alphas',           type=float, nargs='+',
                   default=[1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5],
                   help='Ridge alpha candidates for RidgeCV')
    # Channels
    p.add_argument('--n_jobs',           type=int,   default=1,
                   help='Parallel jobs for RidgeCV (-1 = all cores)')
    p.add_argument('--max_channels',     type=int,   default=None,
                   help='Cap number of channels for speed (None = use all)')
    return p


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Condition-table helpers
# ═══════════════════════════════════════════════════════════════════════════
def load_condition_table(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Expected: Participant, Condition, Run, VideoNumber
    return df


def get_runs_for_subject(df, subject_id, conditions):
    """
    Returns list of (run_number, video_number, condition) tuples
    for the requested subject and condition(s).
    """
    mask = (df['Participant'] == subject_id) & (df['Condition'].isin(conditions))
    sub = df[mask].sort_values('Run')
    return list(zip(sub['Run'], sub['VideoNumber'], sub['Condition']))


def video_filename(video_dir, video_number, condition):
    """
    Map VideoNumber + condition → actual mp4 path.
      forward conditions (1,2) → part<NN>_24Hz.mp4
      backward conditions (3,4) → part<NN>_bw_24Hz.mp4
    """
    tag = 'bw_' if condition in (3, 4) else ''
    name = f'ProjectAttention_movie_part{video_number}_{tag}24Hz.mp4'
    fpath = Path(video_dir) / name
    if not fpath.exists():
        raise FileNotFoundError(f'Video not found: {fpath}')
    return str(fpath)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  MAT-file loading
# ═══════════════════════════════════════════════════════════════════════════
def _inspect_h5(h5_obj, indent=0, max_depth=3):
    """Recursively print HDF5 group/dataset structure."""
    if indent > max_depth:
        return
    prefix = '  ' * indent
    for key in h5_obj.keys():
        item = h5_obj[key]
        if hasattr(item, 'keys'):
            log.info('%s[group] %s', prefix, key)
            _inspect_h5(item, indent + 1, max_depth)
        else:
            log.info('%s[dset]  %s  shape=%s  dtype=%s',
                     prefix, key, item.shape, item.dtype)


def load_meg_mat(mat_path, meg_key='data', prefer_v73=False):
    """
    Load preprocessed MEG data from a FieldTrip-style .mat file.

    Returns
    -------
    meg_data  : dict with keys
        'trial'   – list of arrays, each (n_channels, n_times)  [or None]
        'time'    – list of time vectors                         [or None]
        'label'   – list of channel label strings
        'fsample' – sampling frequency (float)
        'raw'     – the raw loaded object (for debugging)
    """
    mat_path = str(mat_path)
    v73_path = mat_path.replace('.mat', '_v7_3.mat') \
        if not mat_path.endswith('_v7_3.mat') else mat_path
    plain_path = mat_path if not mat_path.endswith('_v7_3.mat') \
        else mat_path.replace('_v7_3.mat', '.mat')

    # Choose which file to open first
    candidates = ([v73_path, plain_path] if prefer_v73
                  else [plain_path, v73_path])

    for path in candidates:
        if not Path(path).exists():
            continue
        log.info('Loading MEG file: %s', path)

        # ── Try HDF5 (v7.3) ──────────────────────────────────────────────
        if path.endswith('_v7_3.mat') or path.endswith('.h5'):
            try:
                return _load_h5(path, meg_key)
            except Exception as e:
                log.warning('HDF5 load failed (%s): %s', path, e)
                continue

        # ── Try legacy scipy.io ──────────────────────────────────────────
        try:
            return _load_scipy(path, meg_key)
        except Exception as e:
            log.warning('scipy.io load failed (%s): %s', path, e)

    raise RuntimeError(f'Could not load MAT file from {candidates}')


def _load_scipy(path, meg_key):
    """Load FieldTrip structure via scipy.io.loadmat."""
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    log.info('scipy.io keys: %s', list(raw.keys()))

    # Navigate to the data struct
    if meg_key in raw:
        ft = raw[meg_key]
    else:
        # Try the first non-private key
        keys = [k for k in raw if not k.startswith('_')]
        log.warning('Key "%s" not found. Available: %s. Using "%s".',
                    meg_key, keys, keys[0])
        ft = raw[keys[0]]

    # FieldTrip struct access via scipy object
    try:
        trial   = list(ft.trial)   if hasattr(ft, 'trial')   else None
        time_v  = list(ft.time)    if hasattr(ft, 'time')     else None
        label   = list(ft.label)   if hasattr(ft, 'label')    else []
        fsample = float(ft.fsample) if hasattr(ft, 'fsample') else 100.0
    except Exception:
        # Maybe it's a plain numpy array (channels x time)
        if hasattr(ft, '__len__'):
            arr = np.array(ft)
            trial  = [arr]
            time_v = [np.arange(arr.shape[-1]) / 100.0]
            label  = [f'ch{i:03d}' for i in range(arr.shape[0])]
            fsample = 100.0
        else:
            raise

    return dict(trial=trial, time=time_v, label=label,
                fsample=fsample, raw=raw)


def _load_h5(path, meg_key):
    """Load FieldTrip structure from HDF5 / v7.3 .mat."""
    out = dict(trial=None, time=None, label=[], fsample=100.0, raw=None)

    with h5py.File(path, 'r') as f:
        log.info('HDF5 top-level keys: %s', list(f.keys()))
        _inspect_h5(f)

        root = f[meg_key] if meg_key in f else f[list(f.keys())[0]]

        # --- fsample ---
        if 'fsample' in root:
            out['fsample'] = float(np.array(root['fsample']).ravel()[0])

        # --- label ---
        if 'label' in root:
            refs = np.array(root['label']).ravel()
            out['label'] = [
                ''.join(chr(c) for c in f[r][()].ravel())
                if isinstance(r, h5py.Reference) else str(r)
                for r in refs
            ]

        # --- trial / time ---
        if 'trial' in root:
            trial_refs = np.array(root['trial']).ravel()
            trials, times = [], []
            for ref in trial_refs:
                arr = np.array(f[ref] if isinstance(ref, h5py.Reference)
                               else root['trial'])
                trials.append(arr)  # shape: (n_ch, n_t) expected
            out['trial'] = trials

        if 'time' in root:
            time_refs = np.array(root['time']).ravel()
            times = []
            for ref in time_refs:
                arr = np.array(f[ref] if isinstance(ref, h5py.Reference)
                               else root['time']).ravel()
                times.append(arr)
            out['time'] = times

        # Fallback: if 'trial' not present but there is a 2-D numeric matrix
        if out['trial'] is None:
            for k in root.keys():
                arr = np.array(root[k])
                if arr.ndim == 2 and arr.dtype.kind in ('f', 'i', 'u'):
                    log.info('Using dataset "%s" as continuous MEG data.', k)
                    out['trial'] = [arr]
                    n_t = arr.shape[-1]
                    out['time'] = [np.arange(n_t) / out['fsample']]
                    break

    return out


def concatenate_runs(meg_dict, run_indices):
    """
    Concatenate a subset of trials (by index) into a single
    (n_channels, n_times) array, returning also a vector of
    run-boundary sample indices (for fold assignment).
    """
    trials = meg_dict['trial']
    selected = [trials[i] for i in run_indices if i < len(trials)]
    if not selected:
        raise ValueError(
            f'No trials found for run_indices={run_indices}. '
            f'Total trials in file: {len(trials)}'
        )
    cat  = np.concatenate(selected, axis=-1)   # (n_ch, total_t)
    boundaries = np.cumsum([0] + [t.shape[-1] for t in selected])
    return cat, boundaries


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Video feature extraction
# ═══════════════════════════════════════════════════════════════════════════
def extract_visual_features(video_path):
    """
    Extract per-frame RMS luminance and frame-to-frame SSIM from a video.

    Returns
    -------
    features : ndarray, shape (n_frames, 2)
        Column 0 = RMS luminance
        Column 1 = SSIM vs previous frame  (first frame gets SSIM=1)
    fps      : float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info('Video: %s  fps=%.1f  frames=%d', video_path, fps, n_frames)

    rms_vals, ssim_vals = [], []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # RMS luminance (root mean square of pixel intensities [0,255])
        rms = np.sqrt(np.mean(gray.astype(np.float32) ** 2))

        # SSIM vs previous frame  (motion / change proxy)
        if prev_gray is None:
            sv = 1.0
        else:
            sv, _ = ssim_fn(prev_gray, gray, full=True)

        rms_vals.append(rms)
        ssim_vals.append(sv)
        prev_gray = gray

    cap.release()
    features = np.column_stack([rms_vals, ssim_vals])
    log.info('Features extracted: %d frames, %d features', *features.shape)
    return features, fps


def resample_features(features, src_fps, dst_fs):
    """
    Resample frame-rate features (src_fps) to MEG sampling frequency (dst_fs).
    Uses polyphase resampling (scipy.signal.resample_poly).

    Returns
    -------
    resampled : ndarray, shape (n_samples_at_dst_fs, n_features)
    """
    from math import gcd
    n_in  = features.shape[0]
    # Compute integer up/down for polyphase resampling
    scale = dst_fs / src_fps
    # Approximate with integer ratio
    denom = 1000
    numer = int(round(scale * denom))
    g     = gcd(numer, denom)
    up, down = numer // g, denom // g

    resampled = np.zeros((int(np.ceil(n_in * up / down)), features.shape[1]))
    for i in range(features.shape[1]):
        resampled[:, i] = signal.resample_poly(features[:, i], up, down)

    log.info('Resampled features: %d → %d samples (%.1f fps → %.1f Hz)',
             n_in, resampled.shape[0], src_fps, dst_fs)
    return resampled


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Temporal lag design matrix
# ═══════════════════════════════════════════════════════════════════════════
def build_lag_matrix(features, lags):
    """
    Build a temporally-lagged (Toeplitz) design matrix.

    Parameters
    ----------
    features : (T, n_feat)
    lags     : array of integer sample lags (can be negative)

    Returns
    -------
    X : (T, n_feat * n_lags)   — zero-padded at boundaries
    """
    T, n_feat = features.shape
    n_lags    = len(lags)
    X = np.zeros((T, n_feat * n_lags), dtype=np.float32)

    for li, lag in enumerate(lags):
        shifted = np.zeros_like(features)
        if lag >= 0:
            shifted[lag:, :] = features[:T - lag, :]
        else:
            shifted[:T + lag, :] = features[-lag:, :]
        X[:, li * n_feat:(li + 1) * n_feat] = shifted

    return X


def make_lags(lag_min_s, lag_max_s, meg_fs):
    """Convert lag range in seconds to integer sample lags."""
    lag_min = int(np.round(lag_min_s * meg_fs))
    lag_max = int(np.round(lag_max_s * meg_fs))
    return np.arange(lag_min, lag_max + 1)


# ═══════════════════════════════════════════════════════════════════════════
#  6.  Ridge regression with k-fold cross-validation
# ═══════════════════════════════════════════════════════════════════════════
def run_encoding_model(X, Y, n_folds, alphas, n_jobs=1):
    """
    Fit a ridge regression encoding model with k-fold CV.

    Parameters
    ----------
    X : (T, n_regressors)   — design matrix (already z-scored or not)
    Y : (T, n_channels)     — MEG data
    n_folds : int
    alphas  : sequence of alpha (regularisation) candidates
    n_jobs  : parallel jobs for RidgeCV

    Returns
    -------
    results : dict
        'r2_per_channel'      (n_channels,)
        'r_per_channel'       (n_channels,)
        'weights'             (n_channels, n_regressors) — from all-data fit
        'best_alpha'          scalar
        'y_pred_cv'           (T, n_channels)  concatenated CV predictions
        'fold_r2'             (n_folds, n_channels)
    """
    T, n_ch = Y.shape


    # ── Handle NaN values in MEG data ──────────────────────────────────
    # Check for NaN channels (entire channel is NaN)
    nan_per_channel = np.isnan(Y).sum(axis=0)
    if (nan_per_channel > 0).any():
        log.warning('Found %d channels with NaN values. Removing them.',
                    (nan_per_channel > 0).sum())
        valid_ch = nan_per_channel == 0
        Y = Y[:, valid_ch]
        n_ch = Y.shape[1]
        log.info('Reduced to %d valid channels', n_ch)

    # Check for NaN timepoints across all channels  
    nan_per_timepoint = np.isnan(Y).any(axis=1)
    if nan_per_timepoint.any():
        valid_t = ~nan_per_timepoint
        log.warning('Found %d timepoints with NaN. Removing them.',
                    nan_per_timepoint.sum())
        X = X[valid_t]
        Y = Y[valid_t]
        T = Y.shape[0]
        log.info('Reduced to %d valid timepoints', T)

    # Replace any remaining NaNs with 0
    if np.isnan(Y).any() or np.isnan(X).any():
        log.warning('Remaining NaNs found. Replacing with 0.')
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)



    # Standardise X (per feature, across time)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    kf     = KFold(n_splits=n_folds, shuffle=False)
    y_pred = np.zeros_like(Y)
    fold_r2 = np.zeros((n_folds, n_ch))

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_sc)):
        log.info('  Fold %d/%d  train=%d  test=%d',
                 fold + 1, n_folds, len(train_idx), len(test_idx))
        ridge = RidgeCV(alphas=alphas, fit_intercept=True,
                        scoring='r2', cv=5)
        ridge.fit(X_sc[train_idx], Y[train_idx])
        y_pred[test_idx] = ridge.predict(X_sc[test_idx])

        for ch in range(n_ch):
            ss_res = np.sum((Y[test_idx, ch] - y_pred[test_idx, ch]) ** 2)
            ss_tot = np.sum((Y[test_idx, ch] - Y[test_idx, ch].mean()) ** 2)
            fold_r2[fold, ch] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Final fit on all data to extract weights
    ridge_all = RidgeCV(alphas=alphas, fit_intercept=True,
                        scoring='r2', cv=5)
    ridge_all.fit(X_sc, Y)
    W = ridge_all.coef_   # (n_ch, n_regressors)

    # Aggregate metrics
    r2_arr = fold_r2.mean(axis=0)
    r_arr  = np.array([
        stats.pearsonr(Y[:, ch], y_pred[:, ch])[0] for ch in range(n_ch)
    ])

    log.info('R² — mean=%.4f  max=%.4f  top-5 channels: %s',
             r2_arr.mean(), r2_arr.max(),
             np.argsort(r2_arr)[-5:][::-1])

    return dict(
        r2_per_channel=r2_arr,
        r_per_channel=r_arr,
        weights=W,
        best_alpha=ridge_all.alpha_,
        y_pred_cv=y_pred,
        fold_r2=fold_r2,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  7.  Reshape weights → TRF (time-lag × feature)
# ═══════════════════════════════════════════════════════════════════════════
def weights_to_trf(weights, n_features, lags):
    """
    weights : (n_channels, n_features * n_lags)  from ridge.coef_
    Returns  : (n_channels, n_lags, n_features)
    """
    n_ch   = weights.shape[0]
    n_lags = len(lags)
    # Column order from build_lag_matrix: lag0_feat0, lag0_feat1, lag1_feat0, ...
    W = weights.reshape(n_ch, n_lags, n_features)
    return W


# ═══════════════════════════════════════════════════════════════════════════
#  8.  Figures
# ═══════════════════════════════════════════════════════════════════════════
FEATURE_NAMES = ['RMS luminance', 'SSIM (frame Δ)']
CONDITION_LABELS = {1: 'Attend forward', 2: 'Unattend forward',
                    3: 'Attend backward', 4: 'Unattend backward'}


def plot_features(features_by_run, fps, meg_fs, out_path):
    """
    Panel figure: feature timeseries for each run.
    Useful for sanity-checking that features look reasonable.
    """
    n_runs = len(features_by_run)
    n_feat = 2
    fig, axes = plt.subplots(n_runs * n_feat, 1,
                             figsize=(14, 3 * n_runs * n_feat),
                             sharex=False)
    axes = np.atleast_1d(axes).ravel()

    ax_i = 0
    for ri, (run_no, vid_no, cond, feat_raw, feat_meg) in \
            enumerate(features_by_run):
        t_vid = np.arange(feat_raw.shape[0]) / fps
        t_meg = np.arange(feat_meg.shape[0]) / meg_fs
        for fi, fname in enumerate(FEATURE_NAMES):
            ax = axes[ax_i]
            ax.plot(t_vid, feat_raw[:, fi], lw=0.8, alpha=0.7,
                    color='steelblue', label='video rate')
            ax.plot(t_meg, feat_meg[:, fi], lw=1.2, alpha=0.9,
                    color='tomato', label='MEG rate')
            ax.set_ylabel(fname, fontsize=8)
            ax.set_title(
                f'Run {run_no}  |  Video {vid_no}  |  '
                f'{CONDITION_LABELS.get(cond, cond)}', fontsize=9)
            if ax_i == 0:
                ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, lw=0.3)
            ax_i += 1

    for ax in axes[ax_i:]:
        ax.set_visible(False)

    fig.supxlabel('Time (s)', y=0.01, fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved feature plot → %s', out_path)


def plot_r2_distribution(r2, labels, out_path):
    """
    Sorted R² bar chart across all channels — good sanity-check figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Sorted bar plot
    order = np.argsort(r2)[::-1]
    ax = axes[0]
    colors = plt.cm.RdYlGn(Normalize(vmin=0, vmax=max(r2.max(), 0.01))(r2[order]))
    ax.bar(np.arange(len(r2)), r2[order], color=colors, edgecolor='none')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Channel (sorted by R²)')
    ax.set_ylabel('Mean cross-validated R²')
    ax.set_title('Encoding model performance — all channels')
    ax.set_xlim(-1, len(r2))
    # Annotate top-5
    for rank, ci in enumerate(order[:5]):
        lbl = labels[ci] if labels else f'ch{ci}'
        ax.annotate(lbl, (rank, r2[ci]),
                    xytext=(rank + 0.5, r2[ci] + 0.002),
                    fontsize=6, rotation=45, color='k')

    # Histogram
    ax2 = axes[1]
    ax2.hist(r2, bins=40, color='steelblue', edgecolor='w', linewidth=0.3)
    ax2.axvline(0, color='k', lw=1)
    ax2.axvline(np.median(r2), color='tomato', lw=1.5,
                label=f'Median = {np.median(r2):.4f}')
    ax2.axvline(r2.mean(), color='gold', lw=1.5,
                label=f'Mean   = {r2.mean():.4f}')
    ax2.set_xlabel('Cross-validated R²')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of R² values')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved R² distribution → %s', out_path)


def plot_trf(trf_weights, lags, meg_fs, labels, r2, out_path, top_n=10):
    """
    TRF (temporal response function) for the top-N channels.
    Shape: trf_weights (n_channels, n_lags, n_features)
    """
    order   = np.argsort(r2)[::-1][:top_n]
    lag_ms  = lags / meg_fs * 1000
    n_feat  = trf_weights.shape[2]

    fig, axes = plt.subplots(top_n, n_feat,
                             figsize=(5 * n_feat, 2.8 * top_n),
                             sharex=True)
    axes = np.atleast_2d(axes)

    for row, ch in enumerate(order):
        lbl = labels[ch] if labels else f'ch{ch}'
        for fi, fname in enumerate(FEATURE_NAMES[:n_feat]):
            ax = axes[row, fi]
            ax.plot(lag_ms, trf_weights[ch, :, fi],
                    lw=1.5, color='steelblue')
            ax.axvline(0, color='k', lw=0.8, ls='--')
            ax.axhline(0, color='gray', lw=0.5)
            ax.set_ylabel('Weight (a.u.)', fontsize=7)
            ax.set_title(f'{lbl}  R²={r2[ch]:.4f}\n{fname}', fontsize=8)
            ax.grid(True, lw=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('Lag (ms)', fontsize=9)

    fig.suptitle(f'TRF for top-{top_n} channels', fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved TRF plot → %s', out_path)


def plot_predicted_vs_actual(Y, Y_pred, meg_fs, labels, r2,
                             out_path, top_n=5, n_seconds=30):
    """
    Overlay of raw and predicted MEG timeseries for the top-N channels.
    Shows only the first n_seconds for clarity.
    """
    order  = np.argsort(r2)[::-1][:top_n]
    n_samp = int(n_seconds * meg_fs)
    t      = np.arange(n_samp) / meg_fs

    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3 * top_n), sharex=True)
    axes = np.atleast_1d(axes)

    for row, ch in enumerate(order):
        lbl = labels[ch] if labels else f'ch{ch}'
        ax  = axes[row]
        ys  = Y[:n_samp, ch]
        yp  = Y_pred[:n_samp, ch]
        # Normalise for display
        ys  = (ys - ys.mean()) / (ys.std() + 1e-8)
        yp  = (yp - yp.mean()) / (yp.std() + 1e-8)
        ax.plot(t, ys, lw=0.8, color='steelblue', alpha=0.9, label='Actual')
        ax.plot(t, yp, lw=1.2, color='tomato',    alpha=0.9, label='Predicted')
        ax.set_ylabel('z-score', fontsize=8)
        ax.set_title(f'{lbl}  R²={r2[ch]:.4f}', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, lw=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.suptitle(f'Actual vs predicted MEG (first {n_seconds}s, top-{top_n} channels)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved pred-vs-actual → %s', out_path)


def plot_fold_reliability(fold_r2, out_path):
    """
    Box plot of per-fold R² distributions: checks for consistency across folds.
    """
    n_folds, n_ch = fold_r2.shape
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [fold_r2[f] for f in range(n_folds)]
    bp   = ax.boxplot(data, patch_artist=True,
                      medianprops=dict(color='k', lw=2))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_folds))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
    ax.axhline(0, color='r', lw=1, ls='--', label='R²=0')
    ax.set_xlabel('Cross-validation fold')
    ax.set_ylabel('R² (all channels)')
    ax.set_title('Per-fold R² distribution — consistency check')
    ax.legend()
    ax.grid(True, axis='y', lw=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved fold reliability → %s', out_path)


def plot_feature_correlation_with_meg(features_meg, Y, meg_fs,
                                      r2, labels, out_path, top_n=20):
    """
    Pearson correlation of each raw feature with each of the top-N channels.
    Sanity check: at short lags, RMS should show some (positive) correlation
    with visual cortex channels.
    """
    order   = np.argsort(r2)[::-1][:top_n]
    n_feat  = features_meg.shape[1]
    corr    = np.zeros((n_feat, top_n))

    for fi in range(n_feat):
        for rank, ch in enumerate(order):
            corr[fi, rank] = stats.pearsonr(
                features_meg[:, fi], Y[:, ch]
            )[0]

    fig, ax = plt.subplots(figsize=(12, 3))
    x       = np.arange(top_n)
    width   = 0.35
    for fi, fname in enumerate(FEATURE_NAMES[:n_feat]):
        offset = (fi - n_feat / 2 + 0.5) * width
        ax.bar(x + offset, corr[fi], width, label=fname, alpha=0.8)

    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels[ch] if labels else f'ch{ch}' for ch in order],
        rotation=45, ha='right', fontsize=7
    )
    ax.set_ylabel('Pearson r')
    ax.set_title(f'Zero-lag feature–MEG correlation (top-{top_n} channels)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved feature-MEG correlation → %s', out_path)


# ═══════════════════════════════════════════════════════════════════════════
#  9.  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = build_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save run config ──────────────────────────────────────────────────
    with open(out_dir / 'run_config.json', 'w') as fh:
        json.dump(vars(args), fh, indent=2)

    # ── Condition table ──────────────────────────────────────────────────
    ct   = load_condition_table(args.condition_table)
    runs = get_runs_for_subject(ct, args.subject, args.conditions)

    if not runs:
        log.error('No runs found for subject %d, conditions %s',
                  args.subject, args.conditions)
        sys.exit(1)

    log.info('Subject %d — %d runs selected: %s',
             args.subject, len(runs), runs)

    # ── Load MEG file ────────────────────────────────────────────────────
    sub_str  = f'sub{args.subject:03d}'
    mat_glob = list(Path(args.data_dir).glob(
        f'{sub_str}_100Hz_badmuscle_badlowfreq_badcomp.mat'
    ))
    if not mat_glob:
        log.error('No MAT file found for %s in %s', sub_str, args.data_dir)
        sys.exit(1)

    meg = load_meg_mat(mat_glob[0], meg_key=args.meg_key,
                       prefer_v73=args.prefer_v73)

    fsample = meg['fsample']
    labels  = list(meg['label']) if meg['label'] else []
    n_trials = len(meg['trial']) if meg['trial'] else 0

    log.info('MEG: %.0f Hz  |  %d channels  |  %d trial(s) / runs',
             fsample, len(labels), n_trials)
    log.info('Channel labels (first 10): %s', labels[:10])

    # Runs are 1-indexed in ConditionTable; trials in .mat are 0-indexed
    # Assumption: trial index = run number - 1
    # *** Adjust this mapping if your .mat ordering differs ***
    run_trial_map = {run_no: run_no - 1 for run_no, _, _ in runs}
    trial_indices = [run_trial_map[r] for r, _, _ in runs]

    log.info('Using trial indices: %s (runs: %s)',
             trial_indices, [r for r, _, _ in runs])

    meg_cat, boundaries = concatenate_runs(meg, trial_indices)
    # meg_cat: (n_channels, n_total_samples)
    log.info('Concatenated MEG: shape=%s  run boundaries=%s',
             meg_cat.shape, boundaries)

    # Optionally cap channels
    if args.max_channels and meg_cat.shape[0] > args.max_channels:
        log.warning('Capping to first %d channels for speed.', args.max_channels)
        meg_cat = meg_cat[:args.max_channels]
        labels  = labels[:args.max_channels]

    # Y: (n_times, n_channels)
    Y = meg_cat.T.astype(np.float32)

    # ── Video features ──────────────────────────────────────────────────
    features_by_run    = []   # for plotting
    all_features_meg   = []   # resampled, per run

    for run_no, vid_no, cond in runs:
        vpath = video_filename(args.video_dir, vid_no, cond)
        log.info('Processing video: %s', vpath)

        feat_raw, fps = extract_visual_features(vpath)
        feat_meg      = resample_features(feat_raw, fps, fsample)

        features_by_run.append((run_no, vid_no, cond, feat_raw, feat_meg))
        all_features_meg.append(feat_meg)

    # Concatenate and align to MEG length
    features_cat = np.concatenate(all_features_meg, axis=0)
    n_meg        = Y.shape[0]
    n_feat       = features_cat.shape[0]

    log.info('Feature samples=%d  MEG samples=%d  Δ=%d',
             n_feat, n_meg, n_feat - n_meg)

    # Crop / pad to match
    if n_feat > n_meg:
        features_cat = features_cat[:n_meg]
    elif n_feat < n_meg:
        pad = np.zeros((n_meg - n_feat, features_cat.shape[1]))
        features_cat = np.vstack([features_cat, pad])

    # ── Lag matrix ──────────────────────────────────────────────────────
    lags = make_lags(args.lag_min, args.lag_max, fsample)
    log.info('TRF window: %.0f ms to %.0f ms  → %d lags',
             args.lag_min * 1000, args.lag_max * 1000, len(lags))

    X = build_lag_matrix(features_cat, lags)
    log.info('Design matrix shape: %s', X.shape)

    # ── Ridge regression ────────────────────────────────────────────────
    log.info('Running ridge regression (%d-fold CV)...', args.n_folds)
    results = run_encoding_model(X, Y, args.n_folds, args.alphas, args.n_jobs)

    r2     = results['r2_per_channel']
    r_vals = results['r_per_channel']
    W      = results['weights']
    Y_pred = results['y_pred_cv']

    # ── Save numerical results ──────────────────────────────────────────
    np.savez(
        out_dir / f'sub{args.subject:03d}_encoding_results.npz',
        r2            = r2,
        r             = r_vals,
        weights       = W,
        fold_r2       = results['fold_r2'],
        lags          = lags,
        feature_names = np.array(FEATURE_NAMES),
        labels        = np.array(labels),
        best_alpha    = results['best_alpha'],
    )
    log.info('Saved results → %s',
             out_dir / f'sub{args.subject:03d}_encoding_results.npz')

    # Summary CSV
    summary = pd.DataFrame({
        'channel': labels if labels else [f'ch{i}' for i in range(len(r2))],
        'r2':      r2,
        'r':       r_vals,
    }).sort_values('r2', ascending=False)
    summary.to_csv(out_dir / f'sub{args.subject:03d}_r2_summary.csv', index=False)

    # ── Figures ─────────────────────────────────────────────────────────
    log.info('Generating figures...')

    trf = weights_to_trf(W, features_cat.shape[1], lags)
    cond_str = '_'.join(str(c) for c in args.conditions)

    plot_features(
        features_by_run, fps, fsample,
        out_dir / f'sub{args.subject:03d}_cond{cond_str}_features.png'
    )
    plot_r2_distribution(
        r2, labels,
        out_dir / f'sub{args.subject:03d}_cond{cond_str}_r2_dist.png'
    )
    plot_trf(
        trf, lags, fsample, labels, r2,
        out_dir / f'sub{args.subject:03d}_cond{cond_str}_trf.png',
        top_n=min(10, len(labels))
    )
    plot_predicted_vs_actual(
        Y, Y_pred, fsample, labels, r2,
        out_dir / f'sub{args.subject:03d}_cond{cond_str}_pred_vs_actual.png',
        top_n=min(5, len(labels))
    )
    plot_fold_reliability(
        results['fold_r2'],
        out_dir / f'sub{args.subject:03d}_cond{cond_str}_fold_reliability.png'
    )
    plot_feature_correlation_with_meg(
        features_cat, Y, fsample, r2, labels,
        out_dir / f'sub{args.subject:03d}_cond{cond_str}_feature_meg_corr.png',
        top_n=min(20, len(labels))
    )

    log.info('=== All done. Results in: %s ===', out_dir)
    log.info('Best alpha: %.2e', results['best_alpha'])
    log.info('R² — mean=%.4f  median=%.4f  max=%.4f',
             r2.mean(), np.median(r2), r2.max())
    log.info('Top-10 channels: %s',
             summary.head(10)[['channel', 'r2']].to_string(index=False))


if __name__ == '__main__':
    main()
