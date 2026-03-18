#!/bin/bash

# SLURM Configuration
#SBATCH --job-name=vjepa2_sliding_analysis
#SBATCH --output=logs/vjepa2_sliding_%A.out
#SBATCH --error=logs/vjepa2_sliding_%A.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=your_email@example.com

# Enable error handling
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================"
echo "V-JEPA2 Sliding Window Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Timestamp: $(date)"
echo "========================================"

# Load CUDA module
echo "Loading CUDA module..."
module load cuda/11.4

# Activate virtual environment
echo "Activating Python environment..."
source /home/predatt/wiesche/generator_env/.venv/bin/activate

# Print environment info
echo ""
echo "Environment Information:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Run the analysis script
echo "Starting V-JEPA2 sliding window analysis..."
python /project/3018078.02/natvidpred_workspace/vidpred_vjepa2_sliding.py

echo ""
echo "========================================"
echo "Job completed!"
echo "Timestamp: $(date)"
echo "========================================"
