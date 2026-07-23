#!/bin/bash
#SBATCH --job-name=logsub_extract
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logsub_extract_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH
/home/alouis/mne_env/bin/python feat_extract_logsub.py \
    --deriv-path /scratch/alouis/dream_bids/derivatives_1000hz/preprocessed-noica \
    --save-path  /scratch/alouis/dream_features_noica_1000hz_logsub \
    --cov-source /scratch/alouis/dream_features_noica_1000hz_overlap \
    --n-jobs     4
