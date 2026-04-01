#!/bin/bash
#SBATCH --job-name=meg_encoding
#SBATCH --output=logs/meg_encoding_sub%a_%j.out
#SBATCH --error=logs/meg_encoding_sub%a_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
##SBATCH --partition=regular          # ← adjust to your HPC partition name
##SBATCH --array=1-72                # ← uncomment to run all subjects as array

# ─────────────────────────────────────────────────────────────────────────────
#  MEG Visual Encoding Analysis — SLURM launcher
#  Runs one subject (or an array of subjects) through meg_encoding_analysis.py
# ─────────────────────────────────────────────────────────────────────────────

# If running as a SLURM array job the subject ID comes from $SLURM_ARRAY_TASK_ID.
# Otherwise set it explicitly below.
# SUBJECT=${SLURM_ARRAY_TASK_ID:-1}
SUBJECT=5

# ── Paths ────────────────────────────────────────────────────────────────────
# PROJECT_DIR="/project/3018078.02/natvidpred_workspace/"                           # ← set this
PROJECT_DIR="/project/3018078.02/MEG_ingmar"                           # ← set this
SCRIPT_DIR="/project/3018078.02/natvidpred_workspace/encoding"
# DATA_DIR="${PROJECT_DIR}/mat_files"                      # .mat files
DATA_DIR="${PROJECT_DIR}"                      # .mat files
VIDEO_DIR="${PROJECT_DIR}"                        # .mp4 files
CONDITION_TABLE="${PROJECT_DIR}/ConditionTable.csv"
OUTPUT_DIR="${PROJECT_DIR}/encoding_results/sub$(printf '%03d' ${SUBJECT})"
SCRIPT="${SCRIPT_DIR}/meg_encoding_analysis.py"

# ── Python / conda environment ───────────────────────────────────────────────
# Adjust to match your HPC module system and conda environment name.
module purge
# module load anaconda3/2023.09          # or: module load miniconda3
module load anaconda3          # or: module load miniconda3
# module load gcc/11.3.0               # sometimes needed for OpenCV

source activate meg_encoding           # conda env name — see environment.yml
# Alternative if using venv:
# source "${PROJECT_DIR}/venv/bin/activate"

# Fix NumPy 1.x/2.x incompatibility: downgrade to NumPy <2
# pip install 'numpy<2' --quiet
pip install 'numpy==1.26.4' --quiet

# ── Experiment settings ──────────────────────────────────────────────────────
CONDITIONS="1 3"          # 1=attend-forward  3=attend-backward
                           # Change to "1 2 3 4" for all conditions

MEG_FS=100                # Hz — from filename (100Hz)
VIDEO_FPS=24              # fps — from filename (24Hz)
LAG_MIN=-0.05             # seconds (negative = anticipatory window)
LAG_MAX=0.50              # seconds
N_FOLDS=5
N_JOBS=${SLURM_CPUS_PER_TASK}

# Ridge alpha grid (log-spaced)
ALPHAS="0.01 0.1 1 10 100 1000 10000 100000"

# ── Make output / log dirs ───────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# ── Log run metadata ─────────────────────────────────────────────────────────
echo "============================================="
echo "  Job:     ${SLURM_JOB_ID}"
echo "  Subject: ${SUBJECT}"
echo "  Node:    $(hostname)"
echo "  Start:   $(date)"
echo "============================================="

# ── Run analysis ─────────────────────────────────────────────────────────────
python "${SCRIPT}" \
    --subject          ${SUBJECT} \
    --conditions       ${CONDITIONS} \
    --data_dir         "${DATA_DIR}" \
    --video_dir        "${VIDEO_DIR}" \
    --condition_table  "${CONDITION_TABLE}" \
    --output_dir       "${OUTPUT_DIR}" \
    --meg_fs           ${MEG_FS} \
    --video_fps        ${VIDEO_FPS} \
    --lag_min          ${LAG_MIN} \
    --lag_max          ${LAG_MAX} \
    --n_folds          ${N_FOLDS} \
    --alphas           ${ALPHAS} \
    --n_jobs           ${N_JOBS}

EXIT_CODE=$?
echo "Finished at $(date) — exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}











# #!/bin/bash
# #SBATCH --job-name=meg_encoding
# #SBATCH --output=logs/meg_encoding_sub%a_%j.out
# #SBATCH --error=logs/meg_encoding_sub%a_%j.err
# #SBATCH --time=04:00:00
# #SBATCH --mem=32G
# #SBATCH --cpus-per-task=8
# ##SBATCH --partition=regular          # ← adjust to your HPC partition name
# ##SBATCH --array=1-72                # ← uncomment to run all subjects as array

# # ─────────────────────────────────────────────────────────────────────────────
# #  MEG Visual Encoding Analysis — SLURM launcher
# #  Runs one subject (or an array of subjects) through meg_encoding_analysis.py
# # ─────────────────────────────────────────────────────────────────────────────

# # If running as a SLURM array job the subject ID comes from $SLURM_ARRAY_TASK_ID.
# # Otherwise set it explicitly below.
# SUBJECT=${SLURM_ARRAY_TASK_ID:-1}

# # ── Paths ────────────────────────────────────────────────────────────────────
# # PROJECT_DIR="/project/3018078.02/natvidpred_workspace/"                           # ← set this
# PROJECT_DIR="/project/3018078.02/MEG_ingmar"                           # ← set this
# SCRIPT_DIR="/project/3018078.02/natvidpred_workspace/encoding"
# # DATA_DIR="${PROJECT_DIR}/mat_files"                      # .mat files
# DATA_DIR="${PROJECT_DIR}"                      # .mat files
# VIDEO_DIR="${PROJECT_DIR}"                        # .mp4 files
# CONDITION_TABLE="${PROJECT_DIR}/ConditionTable.csv"
# OUTPUT_DIR="${PROJECT_DIR}/encoding_results/sub$(printf '%03d' ${SUBJECT})"
# SCRIPT="${SCRIPT_DIR}/meg_encoding_analysis.py"

# # ── Python / conda environment ───────────────────────────────────────────────
# # Adjust to match your HPC module system and conda environment name.
# module purge
# # module load anaconda3/2023.09          # or: module load miniconda3
# module load anaconda3          # or: module load miniconda3
# # module load gcc/11.3.0               # sometimes needed for OpenCV

# source activate meg_encoding           # conda env name — see environment.yml
# # Alternative if using venv:
# # source "${PROJECT_DIR}/venv/bin/activate"

# # ── Experiment settings ──────────────────────────────────────────────────────
# CONDITIONS="1 3"          # 1=attend-forward  3=attend-backward
#                            # Change to "1 2 3 4" for all conditions

# MEG_FS=100                # Hz — from filename (100Hz)
# VIDEO_FPS=24              # fps — from filename (24Hz)
# LAG_MIN=-0.05             # seconds (negative = anticipatory window)
# LAG_MAX=0.50              # seconds
# N_FOLDS=5
# N_JOBS=${SLURM_CPUS_PER_TASK}

# # Ridge alpha grid (log-spaced)
# ALPHAS="0.01 0.1 1 10 100 1000 10000 100000"

# # ── Make output / log dirs ───────────────────────────────────────────────────
# mkdir -p "${OUTPUT_DIR}"
# mkdir -p logs

# # ── Log run metadata ─────────────────────────────────────────────────────────
# echo "============================================="
# echo "  Job:     ${SLURM_JOB_ID}"
# echo "  Subject: ${SUBJECT}"
# echo "  Node:    $(hostname)"
# echo "  Start:   $(date)"
# echo "============================================="

# # ── Run analysis ─────────────────────────────────────────────────────────────
# python "${SCRIPT}" \
#     --subject          ${SUBJECT} \
#     --conditions       ${CONDITIONS} \
#     --data_dir         "${DATA_DIR}" \
#     --video_dir        "${VIDEO_DIR}" \
#     --condition_table  "${CONDITION_TABLE}" \
#     --output_dir       "${OUTPUT_DIR}" \
#     --meg_fs           ${MEG_FS} \
#     --video_fps        ${VIDEO_FPS} \
#     --lag_min          ${LAG_MIN} \
#     --lag_max          ${LAG_MAX} \
#     --n_folds          ${N_FOLDS} \
#     --alphas           ${ALPHAS} \
#     --n_jobs           ${N_JOBS}

# EXIT_CODE=$?
# echo "Finished at $(date) — exit code: ${EXIT_CODE}"
# exit ${EXIT_CODE}
