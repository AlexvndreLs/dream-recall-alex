#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=dream_prep_ica
#SBATCH --array=1-38%6
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/prep_ica_sub-%a_%A.out
#SBATCH --error=logs/prep_ica_sub-%a_%A.err
# ── Relance de la branche ICA uniquement (seuil EOG corr 0.6) ────────────────
# Utilise --branches ica pour ne pas retoucher noica et iclabel déjà corrects.
# s24 (gros sujet, OOM à 64G) : relancer séparément avec --mem=128G :
#     sbatch --array=24 --mem=128G --time=3:00:00 batch_preprocess_ica_only.sh
BIDS_PATH=/home/alouis/scratch/dream_bids
DERIV_ROOT=/home/alouis/scratch/dream_bids/derivatives
SUBJECT=$SLURM_ARRAY_TASK_ID
echo "=== Job array $SLURM_ARRAY_TASK_ID -> sujet $SUBJECT  ($(date)) ==="
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
/home/alouis/mne_env/bin/python preprocess_subject_v3.py "$SUBJECT" \
    --bids-path  "$BIDS_PATH" \
    --deriv-root "$DERIV_ROOT" \
    --branches ica
echo "=== Fini sujet $SUBJECT  ($(date)) ==="
