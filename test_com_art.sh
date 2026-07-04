#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=test_preproc_1000hz_s01
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/alouis/logs_dream/test_preproc_1000hz_s01_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/test_preproc_1000hz_s01_%j.err

source /home/alouis/mne_env/bin/activate
cd /home/alouis/dream-recall-alex

python preprocess_subject_v3.py 1 \
    --bids-path /scratch/alouis/dream_bids \
    --deriv-root /scratch/alouis/dream_bids/derivatives_1000hz \
    --branches noica
