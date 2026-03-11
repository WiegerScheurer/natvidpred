#!/bin/bash
#SBATCH --job-name=mat_to_h5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/project/3018078.02/logs/mat_to_h5_%j.log
#SBATCH --error=/project/3018078.02/logs/mat_to_h5_%j.err

module load matlab

matlab -nodisplay -nosplash <<EOF
run('/project/3018078.02/natvidpred_workspace/scripts/export_to_h5.m');
exit
EOF