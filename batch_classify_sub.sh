#!/bin/bash
#SBATCH --job-name=clf_sub
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-19
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/clf_sub_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

source /home/alouis/mne_env/bin/activate
cd /home/alouis/dream-recall-alex

KEYS=(psd_sub_delta psd_sub_theta psd_sub_alpha psd_sub_sigma psd_sub_beta)
STATES=(S2 SWS REM NREM)

K=$(( SLURM_ARRAY_TASK_ID / 4 ))
S=$(( SLURM_ARRAY_TASK_ID % 4 ))
KEY="${KEYS[$K]}"
STATE="${STATES[$S]}"

echo "combo $SLURM_ARRAY_TASK_ID : $KEY x $STATE"

# --skip-check obligatoire : compute_global_n_trials verifie l'egalite du
# nombre d'epochs entre cov et TOUTES les FEATURE_KEYS de config_v3. Cette
# branche ne contient que cov (symlink) + psd_sub_*, donc le check strict
# signalerait les psd_*/cosp_*/complexite comme absents. n_trials_min est
# neanmoins calcule sur le cov symlinke, donc identique a la branche overlap.
python3 classify.py \
  --save-path        /home/alouis/scratch/dream_features_noica_1000hz_sub \
  --key              "$KEY" \
  --state            "$STATE" \
  --n-jobs           "$SLURM_CPUS_PER_TASK" \
  --n-perm           1000 \
  --n-bootstraps     1000 \
  --checkpoint-every 50 \
  --skip-check
