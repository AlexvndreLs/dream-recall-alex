#!/bin/bash
#SBATCH --job-name=feat_sub
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/feat_sub_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

source /home/alouis/mne_env/bin/activate
cd /home/alouis/dream-recall-alex

python3 feat_extract_sub.py \
  --deriv-path /home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica \
  --save-path  /home/alouis/scratch/dream_features_noica_1000hz_sub \
  --cov-source /home/alouis/scratch/dream_features_noica_1000hz_overlap \
  --n-jobs     "$SLURM_CPUS_PER_TASK"
