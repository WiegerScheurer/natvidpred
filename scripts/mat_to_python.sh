#!/bin/bash
#SBATCH --job-name=mat_to_python
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/3018078.02/logs/mat_to_python_%j.log
#SBATCH --error=/project/3018078.02/logs/mat_to_python_%j.err

module load matlab

matlab -nodisplay -nosplash <<EOF
run('/project/3018078.02/natvidpred_workspace/scripts/export_table.m');
exit
EOF
