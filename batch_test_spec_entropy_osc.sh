#!/bin/bash
#SBATCH --job-name=se_osc_test
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=se_osc_test_%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- activation venv (Fir), identique aux scripts qui tournent ---
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH
# ----------------------------------------------------------------

# Adapter DERIV et SAVE au scratch reel. DERIV = branche noica (reference Arthur directe).
DERIV=/home/alouis/scratch/dream_bids/derivatives/preprocessed-noica
SAVE=/scratch/alouis/dream_features_noica_1000hz_overlap

/home/alouis/mne_env/bin/python test_spec_entropy_osc.py \
    --deriv-path "$DERIV" \
    --save-path  "$SAVE" \
    --n-jobs     "$SLURM_CPUS_PER_TASK" \
    --n-perm     1000 \
    --n-bootstraps 1000
# les 4 etats par defaut (S2, SWS, REM, NREM)
