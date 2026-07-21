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

module load StdEnv/2023 || true
source ~/venvs/mne_env/bin/activate 2>/dev/null || source activate mne_env

# Adapter DERIV et SAVE au scratch reel. DERIV = branche noica (reference Arthur directe).
DERIV=/scratch/alouis/dream_bids/derivatives/preprocessed-noica
SAVE=/scratch/alouis/dream_features_noica_1000hz_overlap

python test_spec_entropy_osc.py \
    --deriv-path "$DERIV" \
    --save-path  "$SAVE" \
    --n-jobs     "$SLURM_CPUS_PER_TASK" \
    --n-perm     1000 \
    --n-bootstraps 1000
# les 4 etats par defaut (S2, SWS, REM, NREM)