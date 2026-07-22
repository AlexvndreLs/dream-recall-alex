#!/bin/bash
#SBATCH --job-name=se_osc_classify
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=se_osc_classify_%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH

DERIV=/scratch/alouis/dream_bids/derivatives_1000hz/preprocessed-noica
SAVE=/scratch/alouis/dream_features_noica_1000hz_overlap

# Classification seule : --skip-extract lit les .npz spec_entropy_osc deja caches.
# Pas de raw charge -> RAM faible, on peut monter n-jobs.
/home/alouis/mne_env/bin/python test_spec_entropy_osc.py \
    --deriv-path "$DERIV" \
    --save-path  "$SAVE" \
    --n-jobs     "$SLURM_CPUS_PER_TASK" \
    --n-perm     1000 \
    --n-bootstraps 1000 \
    --skip-extract
