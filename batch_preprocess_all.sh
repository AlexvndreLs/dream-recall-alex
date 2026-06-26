#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=dream_prep_all
#SBATCH --array=1-38
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=fc30555
#SBATCH --output=/scratch/alouis/logs_dream/prep_all_sub-%a_%A.out
#SBATCH --error=/scratch/alouis/logs_dream/prep_all_sub-%a_%A.err
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

# ── Preprocessing toutes branches (ica + noica + iclabel) ────────────────────
# Patch CAR retiré : référence nez conservée (identique Arthur, rang plein).
# Relancer les 3 branches car toutes utilisaient apply_average_reference.
#
# s24 (gros sujet, OOM à 64G) : exclure du présent array et lancer séparément :
#     sbatch --array=24 --mem=128G --time=4:00:00 batch_preprocess_all.sh
# => array ici : 1-23,25-38 (exclut s24)
#
# Usage normal (tous sujets sauf s24) :
#     sbatch --array=1-23,25-38 batch_preprocess_all.sh
# Puis s24 :
#     sbatch --array=24 --mem=128G --time=4:00:00 batch_preprocess_all.sh

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
    --branches ica noica iclabel

echo "=== Fini sujet $SUBJECT  ($(date)) ==="