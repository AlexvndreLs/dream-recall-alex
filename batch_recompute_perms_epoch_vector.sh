#!/bin/bash
#SBATCH --job-name=recompute_perms_epoch_vec
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-56
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/recompute_perms_epoch_vec-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/recompute_perms_epoch_vec-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ==========================================================================
# Schema EPOCH (replique Arthur, utils.py:103) - features VECTORIELLES.
# Meme principe que batch_recompute_perms_epoch_matrix.sh : reutilise les
# bootstraps existants, ne recalcule que les perms, ecrit en _epochperm.npz.
#
# Array 1-56 : 14 features vectorielles x 4 stades (cf batch_classify_vector.sh)
# ==========================================================================

SAVE_PATH=/scratch/alouis/dream_features_noica_1000hz_overlap   # <-- VERIFIER ce nom avant de lancer

KEYS=(psd_delta psd_theta psd_alpha psd_sigma psd_beta \
      psd_osc_delta psd_osc_theta psd_osc_alpha psd_osc_sigma psd_osc_beta \
      aperiodic perm_entropy higuchi_fd spec_entropy)
STATES=(S2 SWS NREM REM)
KEY_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
STATE_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))
KEY=${KEYS[$KEY_IDX]}
STATE=${STATES[$STATE_IDX]}

echo "=== recompute_perms_epoch (vecteur) key=${KEY} state=${STATE} ==="
echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Start: $(date)"

cd /home/alouis/dream-recall-alex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export FLEXIBLAS_NUM_THREADS=1
source /home/alouis/mne_env/bin/activate

python recompute_perms_epoch_arthur.py \
    --save-path ${SAVE_PATH} \
    --n-jobs $SLURM_CPUS_PER_TASK \
    --n-perm 1000 \
    --checkpoint-every 50 \
    --key ${KEY} --state ${STATE}

echo "End: $(date)"
