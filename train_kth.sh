#!/bin/bash
# --job-name=simvp_gSTA_train_kth
# --output=logs/simvp_train_kth_%A.out   # %A = SLURM job ID
# --error=logs/simvp_train_kth_%A.err
# --time=20:00:00                        # KTH typically takes longer than MovingMNIST
# --partition=gpu
# --gres=gpu:1                           # Request 1 GPU
# --cpus-per-task=8                      # CPU cores for data loading
# --mem=32G
# --mail-type=FAIL
# --mail-user=wieger.scheurer@donders.ru.nl

# =============================================================================
# SimVP-gSTA Training Script for KTH Dataset (OpenSTL framework)
# =============================================================================
# This script trains a SimVP model with gSTA (Gated Spatial-Temporal Attention)
# on the KTH action recognition dataset.
#
# Prerequisites (run once on the login node before submitting):
#   1. cd /project/3018078.02/physical_envs/OpenSTL
#   2. Download KTH dataset (see data verification section below):
#      bash tools/prepare_data/download_kth.sh
# =============================================================================

# Exit immediately if any command fails
set -e

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
module load cuda/11.4
module load gcc/13.3.0

echo "CUDA version:"
nvcc --version

# Move to the OpenSTL project directory and activate its conda environment
cd /project/3018078.02/physical_envs/OpenSTL
source activate OpenSTL

# Limit OpenMP threads to match the allocated CPUs
export OMP_NUM_THREADS=8

echo "Starting SimVP_gSTA training on KTH dataset on $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# -----------------------------------------------------------------------------
# Data verification
# -----------------------------------------------------------------------------
# Check if KTH dataset exists. The KTH dataset should be in data/kth/
# after running the download script
echo "Verifying KTH data..."

# KTH dataset structure:
# data/kth/train/ - contains training video frames
# data/kth/val/ - contains validation video frames
# data/kth/test/ - contains test video frames

[ -d data/kth/train ] && \
[ -d data/kth/val ] && \
[ -d data/kth/test ] && \
echo "Data OK" || { 
  echo "ERROR: KTH data missing. Please download using:"
  echo "  bash tools/prepare_data/download_kth.sh"
  exit 1
}

# Count number of training sequences to verify data is present
TRAIN_COUNT=$(find data/kth/train -type d | wc -l)
echo "Found $TRAIN_COUNT training sequences"

# -----------------------------------------------------------------------------
# Training
# =============================================================================

echo "Launching training..."

python tools/train.py \
    -d kth \
    -c configs/kth/simvp/SimVP_gSTA.py \
    --ex_name kth_simvp_gsta_pretrained \
    --gpus 0 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 1e-3 \
    --epoch 200

# -----------------------------------------------------------------------------
# Verify output
# =============================================================================
echo "✅ Training complete on $(date)"
echo "Weights saved to: work_dirs/kth_simvp_gsta_pretrained/checkpoints/"
ls -la work_dirs/kth_simvp_gsta_pretrained/checkpoints/
