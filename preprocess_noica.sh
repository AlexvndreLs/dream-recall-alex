#!/bin/bash
#SBATCH --account=rrg-kjerbi
#SBATCH --job-name=dream_prep_noica250
#SBATCH --array=1-23,25-38
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=fc30555
#SBATCH --output=/scratch/alouis/logs_dream/prep_noica_sub-%a_%A.out
#SBATCH --error=/scratch/alouis/logs_dream/prep_noica_sub-%a_%A.err
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

# Regenere UNIQUEMENT la branche noica a 250Hz dans derivatives/ (250).
# PREREQUIS config_v3.py : DECIMATE=True ET SFREQ_TARGET=250.0 (les DEUX).
# s24 exclu de l'array (OOM 64G) -> lancer separement :
#   sbatch --array=24 --mem=128G --time=2:30:00 preprocess_noica.sh

BIDS_PATH=/home/alouis/scratch/dream_bids
DERIV_ROOT=/home/alouis/scratch/dream_bids/derivatives_250hz_dl   # 250Hz DL isole (pas d'overwrite)
SUBJECT=$SLURM_ARRAY_TASK_ID

echo "=== Job $SLURM_ARRAY_TASK_ID -> sujet $SUBJECT ($(date)) ==="
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

/home/alouis/mne_env/bin/python preprocess_subject_v3.py "$SUBJECT" \
    --bids-path  "$BIDS_PATH" \
    --deriv-root "$DERIV_ROOT" \
    --branches noica

echo "=== Fini sujet $SUBJECT ($(date)) ==="
