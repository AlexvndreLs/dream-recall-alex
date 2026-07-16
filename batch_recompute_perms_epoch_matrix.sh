#!/bin/bash
#SBATCH --job-name=recompute_perms_epoch_mat
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-24
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=32
#SBATCH --output=/scratch/alouis/logs_dream/recompute_perms_epoch_mat-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/recompute_perms_epoch_mat-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ==========================================================================
# Schema EPOCH (rreplique Arthur, utils.py:103) - features MATRICIELLES.
# Reutilise les bootstraps deja calcules par le run --perm-scheme subject
# (classify.py) : AUCUN recalcul des 1000 bootstraps, seules les perms sont
# refaites. Ecrit dans {key}_{state}_epochperm.npz (meme dossier results/,
# le fichier subject original n'est JAMAIS touche).
#
# PREREQUIS : results/{key}_{state}.npz doit deja exister (run subject termine).
#
# Array 1-24 : 6 features matricielles x 4 stades (cf batch_classify_matrix.sh)
# ==========================================================================

SAVE_PATH=/scratch/alouis/dream_features_noica_1000hz   # <-- VERIFIER ce nom avant de lancer

KEYS=(cov cosp_delta cosp_theta cosp_alpha cosp_sigma cosp_beta)
STATES=(S2 SWS NREM REM)
KEY_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
STATE_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))
KEY=${KEYS[$KEY_IDX]}
STATE=${STATES[$STATE_IDX]}

echo "=== recompute_perms_epoch (matrice) key=${KEY} state=${STATE} ==="
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
