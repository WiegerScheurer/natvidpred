#!/bin/bash
# preflight_k400_openstl.sh
# Validate OpenSTL + Kinetics-400 setup before launching GPU training

set -euo pipefail

# -----------------------------
# Defaults (override via flags)
# -----------------------------
OPENSTL_ROOT="/project/3018078.02/physical_envs/OpenSTL"
DATA_ROOT=""
CONDA_ENV="OpenSTL"
CONFIG=""
DATASET_FLAG=""   # auto: kinetics400 -> kinetics fallback

usage() {
  cat <<EOF
Usage:
  bash preflight_k400_openstl.sh [options]

Options:
  --openstl-root PATH   Path to OpenSTL repo (default: ${OPENSTL_ROOT})
  --data-root PATH      Path to Kinetics-400 root (default: <openstl-root>/data/kinetics400)
  --conda-env NAME      Conda env name (default: ${CONDA_ENV})
  --config PATH         Explicit config file path (optional; auto-detect if omitted)
  --dataset FLAG        Dataset flag for train.py (kinetics400 or kinetics) (optional; auto-detect if omitted)
  -h, --help            Show this help
EOF
}

# -----------------------------
# Arg parsing
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --openstl-root) OPENSTL_ROOT="$2"; shift 2 ;;
    --data-root)    DATA_ROOT="$2"; shift 2 ;;
    --conda-env)    CONDA_ENV="$2"; shift 2 ;;
    --config)       CONFIG="$2"; shift 2 ;;
    --dataset)      DATASET_FLAG="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done


if [[ -z "${DATA_ROOT}" ]]; then
  candidates=(
    "${OPENSTL_ROOT}/data/kinetics400"
    "${OPENSTL_ROOT}/data/kinetics"
    "${OPENSTL_ROOT}/data/k400"
    "/project/3018078.02/datasets/kinetics400"
    "/project/3018078.02/datasets/kinetics"
  )
  for d in "${candidates[@]}"; do
    if [[ -d "$d" ]]; then
      DATA_ROOT="$d"
      break
    fi
  done
fi

[[ -n "${DATA_ROOT}" && -d "${DATA_ROOT}" ]] || {
  echo "❌ Could not find K400 data root automatically."
  echo "Pass it explicitly with: --data-root /path/to/kinetics400"
  exit 1
}
# if [[ -z "${DATA_ROOT}" ]]; then
#   DATA_ROOT="${OPENSTL_ROOT}/data/kinetics400"
# fi

pass() { echo "✅ $1"; }
warn() { echo "⚠️  $1"; }
fail() { echo "❌ $1"; exit 1; }

echo "=== OpenSTL K400 Preflight ==="
echo "OpenSTL root : ${OPENSTL_ROOT}"
echo "Data root    : ${DATA_ROOT}"
echo "Conda env    : ${CONDA_ENV}"
echo

# -----------------------------
# Basic filesystem checks
# -----------------------------
[[ -d "${OPENSTL_ROOT}" ]] || fail "OpenSTL root not found: ${OPENSTL_ROOT}"
[[ -f "${OPENSTL_ROOT}/tools/train.py" ]] || fail "Missing tools/train.py in ${OPENSTL_ROOT}"
[[ -d "${DATA_ROOT}" ]] || fail "Data root not found: ${DATA_ROOT}"
pass "OpenSTL repo and K400 data root exist"

# -----------------------------
# Activate env
# -----------------------------
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" || fail "Could not activate conda env: ${CONDA_ENV}"
  pass "Conda env activated: ${CONDA_ENV}"
else
  fail "conda command not found in PATH"
fi

# -----------------------------
# Python package checks
# -----------------------------
python - <<'PY'
import importlib.util, sys
mods = ["torch", "numpy", "einops", "timm"]
optional = ["decord", "av", "cv2"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING_REQUIRED=" + ",".join(missing))
    sys.exit(2)
missing_optional = [m for m in optional if importlib.util.find_spec(m) is None]
print("MISSING_OPTIONAL=" + ",".join(missing_optional))
PY
status=$?
if [[ $status -ne 0 ]]; then
  fail "Missing required Python packages. Install them in ${CONDA_ENV}."
fi
pass "Required Python packages are present"

# -----------------------------
# Auto-detect config if needed
# -----------------------------
if [[ -z "${CONFIG}" ]]; then
  candidates=(
    "${OPENSTL_ROOT}/configs/kinetics400/simvp/SimVP_gSTA.py"
    "${OPENSTL_ROOT}/configs/kinetics/simvp/SimVP_gSTA.py"
    "${OPENSTL_ROOT}/configs/kinetics400/simvp/SimVP.py"
    "${OPENSTL_ROOT}/configs/kinetics/simvp/SimVP.py"
  )
  for c in "${candidates[@]}"; do
    if [[ -f "${c}" ]]; then
      CONFIG="${c}"
      break
    fi
  done
fi
[[ -n "${CONFIG}" ]] || fail "Could not auto-detect a Kinetics SimVP config. Pass --config explicitly."
[[ -f "${CONFIG}" ]] || fail "Config file not found: ${CONFIG}"
pass "Config found: ${CONFIG}"

# -----------------------------
# Auto-detect dataset flag
# -----------------------------
if [[ -z "${DATASET_FLAG}" ]]; then
  if grep -qi "kinetics400" "${CONFIG}"; then
    DATASET_FLAG="kinetics400"
  else
    DATASET_FLAG="kinetics"
  fi
fi
if [[ "${DATASET_FLAG}" != "kinetics400" && "${DATASET_FLAG}" != "kinetics" ]]; then
  fail "Invalid --dataset value: ${DATASET_FLAG} (use kinetics400 or kinetics)"
fi
pass "Dataset flag selected: ${DATASET_FLAG}"

# -----------------------------
# Annotation checks
# -----------------------------
ann_candidates=(
  "${DATA_ROOT}/train.csv"
  "${DATA_ROOT}/val.csv"
  "${DATA_ROOT}/train.txt"
  "${DATA_ROOT}/val.txt"
  "${DATA_ROOT}/annotations/train.csv"
  "${DATA_ROOT}/annotations/val.csv"
  "${DATA_ROOT}/annotations/train.txt"
  "${DATA_ROOT}/annotations/val.txt"
)

train_ann=""
val_ann=""
for f in "${ann_candidates[@]}"; do
  base="$(basename "$f")"
  if [[ -f "$f" ]]; then
    [[ -z "${train_ann}" && "$base" == train.* ]] && train_ann="$f"
    [[ -z "${val_ann}"   && "$base" == val.*   ]] && val_ann="$f"
  fi
done

[[ -n "${train_ann}" ]] || fail "No train annotation file found in ${DATA_ROOT} (or annotations/)"
[[ -n "${val_ann}" ]] || warn "No val annotation file found (training may still start, but validation can fail)"
pass "Annotation files detected"

# -----------------------------
# Video file presence checks
# -----------------------------
video_probe=$(find "${DATA_ROOT}" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.webm" -o -iname "*.mkv" \) | head -n 1 || true)
[[ -n "${video_probe}" ]] || warn "No video files found under ${DATA_ROOT}; if you use extracted frames, this may be expected"
if [[ -n "${video_probe}" ]]; then
  pass "Found at least one video file: ${video_probe}"
fi

# -----------------------------
# Annotation path resolvability check
# -----------------------------
python - "${DATA_ROOT}" "${train_ann}" <<'PY'
import csv, os, sys
data_root = sys.argv[1]
ann = sys.argv[2]
checked = 0
resolved = 0

def parse_line(raw):
    raw = raw.strip()
    if not raw:
        return None
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        return parts[0]
    return raw.split()[0]

with open(ann, "r", encoding="utf-8") as f:
    for line in f:
        p = parse_line(line)
        if p is None:
            continue
        checked += 1
        cand = p if os.path.isabs(p) else os.path.join(data_root, p)
        if os.path.exists(cand):
            resolved += 1
        if checked >= 50:
            break

print(f"Checked {checked} annotation rows; resolved {resolved} paths")
if checked == 0:
    print("ERROR: annotation appears empty")
    sys.exit(2)
if resolved == 0:
    print("ERROR: none of sampled annotation paths resolved on disk")
    sys.exit(3)
PY
status=$?
if [[ $status -ne 0 ]]; then
  fail "Annotation content check failed (path mismatch between annotation and DATA_ROOT)."
fi
pass "Annotation entries map to existing files/directories"

# -----------------------------
# Final summary + ready command
# -----------------------------
echo
echo "=== PRECHECK PASSED ==="
echo "Suggested train command:"
echo "python tools/train.py \\"
echo "  -d ${DATASET_FLAG} \\"
echo "  -c ${CONFIG} \\"
echo "  --ex_name k400_simvp_gsta \\"
echo "  --gpus 0 \\"
echo "  --batch_size 8 \\"
echo "  --num_workers 8 \\"
echo "  --lr 1e-3 \\"
echo "  --epoch 200"