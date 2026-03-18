#!/bin/bash
#SBATCH --job-name=vjepa_sliding_window
#SBATCH --output=logs/vjepa_bulk_%A.out
#SBATCH --error=logs/vjepa_bulk_%A.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END

# Load modules (adjust versions to your HPC's availability)
module load cuda/11.4

# Activate your specific environment
source /home/predatt/wiesche/generator_env/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the script
python /project/3018078.02/natvidpred_workspace/vidpred_vjepa2_bulk.py