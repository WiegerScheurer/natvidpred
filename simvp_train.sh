#!/bin/bash
# --job-name=simvp_gSTA_train
# --output=logs/simvp_train_%A.out   # %A = SLURM job ID
# --error=logs/simvp_train_%A.err
# --time=10:00:00                    # 50 epochs takes ~1.3h; 4h gives comfortable headroom
# --partition=gpu
# --gres=gpu:1                       # Request 1 GPU
# --cpus-per-task=8                  # CPU cores for data loading (matches --num_workers below)
# --mem=32G
# --mail-type=FAIL
# --mail-user=wieger.scheurer@donders.ru.nl

# =============================================================================
# SimVP-gSTA Training Script for MovingMNIST (OpenSTL framework)
# =============================================================================
# This script trains a SimVP model with gSTA (Gated Spatial-Temporal Attention)
# on the MovingMNIST dataset as a sanity check / baseline run.
#
# Expected runtime: ~1.3 hours for 50 epochs on a single A100 GPU
# Expected output:  work_dirs/mmnist_simvp_gsta_pretrained/saved/epoch_best.pth
#
# Prerequisites (run once on the login node before submitting):
#   1. cd /project/3018078.02/physical_envs/OpenSTL
#   2. wget -O data/moving_mnist/train-images-idx3-ubyte.gz \
#        https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
#   (yann.lecun.com returns a broken 122-byte HTML page instead of the real file,
#    so we use the Google mirror instead)
# =============================================================================

# Exit immediately if any command fails
set -e

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
# Load CUDA and GCC modules. Note: we load cuda/11.4 as the system module,
# but PyTorch in this environment was compiled against CUDA 12.8 and will use
# that runtime automatically — the module load is mainly for nvcc/toolchain
# compatibility.
module load cuda/11.4
module load gcc/13.3.0

echo "CUDA version:"
nvcc --version

# Move to the OpenSTL project directory and activate its conda environment
cd /project/3018078.02/physical_envs/OpenSTL
source activate OpenSTL

# Limit OpenMP threads to match the allocated CPUs, preventing accidental
# over-subscription of CPU resources on shared nodes
export OMP_NUM_THREADS=8

echo "Starting SimVP_gSTA training on $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# -----------------------------------------------------------------------------
# Data verification
# -----------------------------------------------------------------------------
# We download data once on the login node rather than in the SLURM job, because:
#   - Downloading inside a GPU job wastes expensive GPU-hours
#   - Interrupted downloads leave corrupted files that cause silent training errors
#   - yann.lecun.com (the original source) currently returns a broken HTML page
#     instead of the actual gz file, so downloads must use the Google mirror
#
# This check verifies the training file is present and >1MB (the real file is
# ~9.5MB; the broken HTML version is only 122 bytes).
echo "Verifying data..."
TRAIN_SIZE=$(stat -c%s data/moving_mnist/train-images-idx3-ubyte.gz 2>/dev/null || echo 0)
[ -f data/moving_mnist/mnist_test_seq.npy ] && \
[ "$TRAIN_SIZE" -gt 1000000 ] && \
echo "Data OK" || { echo "ERROR: data missing or corrupted. Check data/moving_mnist/ and re-run the login-node download step."; exit 1; }

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
echo "Launching training..."

    # NOTE: --gpus refers to the GPU *device index*, not the number of GPUs.
    # SLURM always assigns the allocated GPU as device index 0 via
    # CUDA_VISIBLE_DEVICES, so this must always be 0 (not 1) in single-GPU jobs.

python tools/train.py \
    -d mmnist \
    -c configs/mmnist/simvp/SimVP_gSTA.py \
    --ex_name mmnist_simvp_gsta_pretrained \
    --gpus 0 \
    --batch_size 32 \
    --num_workers 4 \   # DataLoader workers; keep <= cpus-per-task
    --lr 1e-3 \
    --epoch 200          # Sanity check run; full training uses 200 epochs (~5.2h)

# -----------------------------------------------------------------------------
# Verify output
# -----------------------------------------------------------------------------
echo "✅ Training complete on $(date)"
# echo "Weights saved to: work_dirs/mmnist_simvp_gsta_pretrained/saved/epoch_best.pth"
echo "Weights saved to: work_dirs/mmnist_simvp_gsta_pretrained/checkpoints/best.cpkt" #of iets dergelijks
ls -la work_dirs/mmnist_simvp_gsta_pretrained/saved/




#!/bin/bash
# # --job-name=simvp_gsta_k400
# # --output=logs/simvp_k400_%A.out
# # --error=logs/simvp_k400_%A.err
# # --time=24:00:00
# # --partition=gpu
# # --gres=gpu:1
# # --cpus-per-task=8
# # --mem=64G
# # --mail-type=FAIL
# # --mail-user=wieger.scheurer@donders.ru.nl

# set -euo pipefail

# module load cuda/11.4
# module load gcc/13.3.0

# cd /project/3018078.02/physical_envs/OpenSTL
# source activate OpenSTL

# export OMP_NUM_THREADS=8
# mkdir -p logs

# echo "Starting SimVP-gSTA Kinetics-400 training on $(date)"
# echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# nvcc --version

# # -----------------------------------------------------------------------------
# # Dataset + config (adjust these paths to your local OpenSTL setup)
# # -----------------------------------------------------------------------------
# DATA_ROOT="data/kinetics400"
# EX_NAME="k400_simvp_gsta"

# # Try common config locations used in OpenSTL forks
# if [ -f configs/kinetics400/simvp/SimVP_gSTA.py ]; then
#     CFG="configs/kinetics400/simvp/SimVP_gSTA.py"
# elif [ -f configs/kinetics/simvp/SimVP_gSTA.py ]; then
#     CFG="configs/kinetics/simvp/SimVP_gSTA.py"
# else
#     echo "ERROR: Could not find SimVP_gSTA Kinetics config."
#     echo "Looked for:"
#     echo "  configs/kinetics400/simvp/SimVP_gSTA.py"
#     echo "  configs/kinetics/simvp/SimVP_gSTA.py"
#     exit 1
# fi

# # Basic data presence check (customize to your exact K400 layout)
# if [ ! -d "${DATA_ROOT}" ]; then
#     echo "ERROR: ${DATA_ROOT} not found."
#     exit 1
# fi

# echo "Using config: ${CFG}"
# echo "Using data root: ${DATA_ROOT}"

# # -----------------------------------------------------------------------------
# # Training
# # -----------------------------------------------------------------------------
# python tools/train.py \
#     -d kinetics400 \
#     -c "${CFG}" \
#     --ex_name "${EX_NAME}" \
#     --gpus 0 \
#     --batch_size 8 \
#     --num_workers 8 \
#     --lr 1e-3 \
#     --epoch 200

# echo "Training complete on $(date)"
# echo "Checkpoints:"
# ls -lah "work_dirs/${EX_NAME}/" || true
