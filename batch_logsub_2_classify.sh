#!/bin/bash
#SBATCH --job-name=logsub_clf
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logsub_clf_%A_%a.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH
SAVE=/scratch/alouis/dream_features_noica_1000hz_logsub
KEYS=(psd_logsub_delta psd_logsub_theta psd_logsub_alpha psd_logsub_sigma psd_logsub_beta)
STATES=(S2 SWS REM NREM)
K=$(( SLURM_ARRAY_TASK_ID / 4 ))
S=$(( SLURM_ARRAY_TASK_ID % 4 ))
KEY="${KEYS[$K]}"
STATE="${STATES[$S]}"
echo "combo $SLURM_ARRAY_TASK_ID : $KEY x $STATE"
/home/alouis/mne_env/bin/python classify.py \
    --save-path        "$SAVE" \
    --key              "$KEY" \
    --state            "$STATE" \
    --n-jobs           "$SLURM_CPUS_PER_TASK" \
    --n-perm           1000 \
    --n-bootstraps     1000 \
    --checkpoint-every 50 \
    --skip-check
