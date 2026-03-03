#!/bin/bash
#SBATCH --job-name=simvp_k400_train
#SBATCH --output=logs/simvp_k400_%A.out
#SBATCH --error=logs/simvp_k400_%A.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wieger.scheurer@donders.ru.nl

set -euo pipefail

module load cuda/11.4
module load gcc/13.3.0

OPENSTL_ROOT="/project/3018078.02/physical_envs/OpenSTL"
CONDA_ENV="OpenSTL"

cd "${OPENSTL_ROOT}"
source activate "${CONDA_ENV}"

mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "Starting SimVP Kinetics training on $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvcc --version

# -----------------------------------------------------------------------------
# Dataset + config (override via: sbatch --export=ALL,DATA_ROOT=/path,EX_NAME=name)
# -----------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-data/kinetics400}"
EX_NAME="${EX_NAME:-k400_simvp_baseline}"

# Your repo has configs/kinetics/SimVP.py; keep fallback for other layouts.
if [ -f configs/kinetics/SimVP.py ]; then
    CFG="configs/kinetics/SimVP.py"
elif [ -f configs/kinetics400/SimVP.py ]; then
    CFG="configs/kinetics400/SimVP.py"
else
    echo "ERROR: Could not find Kinetics SimVP config."
    echo "Looked for:"
    echo "  configs/kinetics/SimVP.py"
    echo "  configs/kinetics400/SimVP.py"
    exit 1
fi

# Dataset flag can differ across OpenSTL versions.
if python tools/train.py -h 2>&1 | grep -q -- "kinetics400"; then
    DATASET_FLAG="kinetics400"
else
    DATASET_FLAG="kinetics"
fi

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: ${DATA_ROOT} not found."
    echo "Tip: pass your actual path with:"
    echo "  sbatch --export=ALL,DATA_ROOT=/absolute/path/to/kinetics400 simvp_train.sh"
    exit 1
fi

TRAIN_ANN=""
for f in \
  "${DATA_ROOT}/train.csv" \
  "${DATA_ROOT}/train.txt" \
  "${DATA_ROOT}/annotations/train.csv" \
  "${DATA_ROOT}/annotations/train.txt"
do
  [ -f "$f" ] && TRAIN_ANN="$f" && break
done

if [ -z "${TRAIN_ANN}" ]; then
    echo "ERROR: no train annotation file found under ${DATA_ROOT}"
    echo "Expected train.csv/train.txt in root or annotations/."
    exit 1
fi

echo "Using config: ${CFG}"
echo "Using data root: ${DATA_ROOT}"
echo "Using dataset flag: ${DATASET_FLAG}"
echo "Found train annotation: ${TRAIN_ANN}"

python - <<'PY'
import importlib.util
required = ["torch", "numpy", "einops", "timm"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing required python packages: " + ", ".join(missing))
print("Python dependency check passed")
PY

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
python tools/train.py \
    -d "${DATASET_FLAG}" \
    -c "${CFG}" \
    --ex_name "${EX_NAME}" \
    --gpus 0 \
    --batch_size 8 \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --lr 1e-3 \
    --epoch 200

echo "Training complete on $(date)"
echo "Checkpoints/output: work_dirs/${EX_NAME}/"
ls -lah "work_dirs/${EX_NAME}/" || true
