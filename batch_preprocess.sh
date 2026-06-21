#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=dream_prep
#SBATCH --array=1-38%6
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/prep_sub-%a_%A.out
#SBATCH --error=logs/prep_sub-%a_%A.err

# ── Batch preprocessing des 38 sujets (array SLURM) ──────────────────────────
# Un job par sujet : $SLURM_ARRAY_TASK_ID = numéro de sujet (1..38).
# %6 : max 6 jobs simultanés -> ne monopolise pas le cluster.
# Chaque job fitte les 3 branches (ica Picard + noica + iclabel) et sauve les
# ICA dans derivatives/ica/. Sujets 21-22 preprocessés normalement (exclus en
# aval). Lancer depuis ~/dream-recall-alex :
#     mkdir -p logs && sbatch batch_preprocess.sh

BIDS_PATH=/home/alouis/scratch/dream_bids
DERIV_ROOT=/home/alouis/scratch/dream_bids/derivatives
SUBJECT=$SLURM_ARRAY_TASK_ID

echo "=== Job array $SLURM_ARRAY_TASK_ID -> sujet $SUBJECT  ($(date)) ==="

source /home/alouis/mne_env/bin/activate

# threads BLAS/OMP alignés sur l'allocation CPU (Picard/numpy)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python preprocess_subject_v3.py "$SUBJECT" \
    --bids-path  "$BIDS_PATH" \
    --deriv-root "$DERIV_ROOT"

echo "=== Fini sujet $SUBJECT  ($(date)) ==="
