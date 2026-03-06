#!/bin/bash
#SBATCH --job-name=vjepa2_decoder_train
#SBATCH --output=logs/vjepa2_decoder_%A.out
#SBATCH --error=logs/vjepa2_decoder_%A.err
#SBATCH --time=16:00:00                   # Preprocessing (~50min) + 40 epochs (~14-15h)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                       # 1 A100 GPU
#SBATCH --cpus-per-task=8                  # CPU cores for video loading
#SBATCH --mem=64G                          # Pre-encoded clips fit in RAM
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wieger.scheurer@donders.ru.nl

# =============================================================================
# V-JEPA 2 Pixel Decoder Training
# =============================================================================
# Trains a convolutional decoder to map V-JEPA 2 patch tokens → pixel frames.
#
# Expected runtime: ~1-2 hours per 40 epochs on one A100 GPU
# Expected output: decoder_checkpoints/vjepa2_decoder_best.pt
#                  decoder_checkpoints/vjepa2_decoder_final.pt
#
# Prerequisites (run once on login node before submitting):
#   - Video clips must exist at /project/3018078.02/MEG_ingmar/shorts/*.mp4
#   - HuggingFace cache dir (usually ~/.cache/huggingface) has space (~8GB)
#
# Memory note: The script pre-encodes all video clips to RAM at init.
#   With CLIPS_PER_VIDEO=20 and multiple videos:
#   - Each clip stores ~32 (token, frame) pairs
#   - Each pair is ~(576*1408*4 + 3*384*384*4) bytes ≈ 3.3 MB
#   - 6 videos × 20 clips × 32 pairs × 3.3 MB ≈ 12.7 GB
#   64GB is comfortably safe; reduce to 32GB if failed.
# =============================================================================

set -e

module load cuda/11.4
module load gcc/13.3.0

echo "CUDA version:"
nvcc --version
echo ""

# Activate environment
# source /home/predatt/wiesche/physical_envs/generator_env/.venv/bin/activate
cd /home/predatt/wiesche/generator_env
source .venv/bin/activate
echo "Activated Python environment: $(which python)"
echo ""


# Create logs directory
mkdir -p logs

echo "Starting V-JEPA 2 decoder training..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $SLURM_GPUS_PER_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""
echo "Assigned GPU details:"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
echo ""

cd /project/3018078.02/natvidpred_workspace

python train_vjepa2_decoder.py

echo ""
echo "Training complete. Checkpoints saved to: decoder_checkpoints/"
echo "Job completed at $(date)"



# --partition=gpu
# --gres=gpu:nvidia_a100-pcie-40gb:1