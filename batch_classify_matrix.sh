#!/bin/bash
#SBATCH --job-name=classify_matrix
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-24
#SBATCH --time=18:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=32
#SBATCH --output=/scratch/alouis/logs_dream/classify_matrix-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/classify_matrix-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Batch par combo matriciel (cov/cosp_* × 4 stades × 3 branches).
# 1 job = 1 (key, state) = 1 feature × 1 stade.
# Checkpoint toutes les 50 itérations -> reprise après timeout sans repartir de 0.
#
# Array 1-24 : 6 features matricielles × 4 stades
#   features : cov, cosp_delta, cosp_theta, cosp_alpha, cosp_sigma, cosp_beta
#   stades   : S2, SWS, NREM, REM
#
# Pour lancer sur 1 seule branche :
#   sbatch --array=1-24 batch_classify_matrix.sh 1    (ica)
#   sbatch --array=1-24 batch_classify_matrix.sh 2    (noica)
#   sbatch --array=1-24 batch_classify_matrix.sh 3    (iclabel)
# Ou les 3 branches en parallèle :
#   for b in 1 2 3; do sbatch --export=BRANCH=$b batch_classify_matrix.sh; done

SAVE_ROOT=/home/alouis/scratch
BRANCH=${BRANCH:-2}   # branche par défaut : ica

case $BRANCH in
    1) SAVE=dream_features         ;;
    2) SAVE=dream_features_noica_1000hz_overlap   ;;
    3) SAVE=dream_features_iclabel ;;
esac

# Mapping array -> (key, state)
KEYS=(cov cosp_delta cosp_theta cosp_alpha cosp_sigma cosp_beta)
STATES=(S2 SWS NREM REM)

# index 1-based : KEY_IDX=0..5, STATE_IDX=0..3
KEY_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
STATE_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))
KEY=${KEYS[$KEY_IDX]}
STATE=${STATES[$STATE_IDX]}

echo "=== classify_matrix branch=${SAVE} key=${KEY} state=${STATE} ==="
echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Start: $(date)"

cd /home/alouis/dream-recall-alex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export FLEXIBLAS_NUM_THREADS=1

/home/alouis/mne_env/bin/python classify.py \
    --save-path       ${SAVE_ROOT}/${SAVE} \
    --n-jobs          $SLURM_CPUS_PER_TASK \
    --n-perm          1000 \
    --n-bootstraps    1000 \
    --checkpoint-every 50 \
    --key             ${KEY} \
    --state           ${STATE} \
    --skip-check

echo "End: $(date)"
