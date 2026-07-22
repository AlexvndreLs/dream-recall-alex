#!/bin/bash
#SBATCH --job-name=se_osc_extract
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=se_osc_extract_%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH

DERIV=/scratch/alouis/dream_bids/derivatives_1000hz/preprocessed-noica
SAVE=/scratch/alouis/dream_features_noica_1000hz_overlap

# Extraction seule. n-perm 0 + on s'arrete avant la classif via --skip-extract absent.
# n-jobs-extract bas : chaque worker charge un raw full-night 1000Hz.
/home/alouis/mne_env/bin/python test_spec_entropy_osc.py \
    --deriv-path "$DERIV" \
    --save-path  "$SAVE" \
    --n-jobs-extract 4 \
    --n-jobs     1 \
    --n-perm     0 \
    --n-bootstraps 1 \
    --extract-only

