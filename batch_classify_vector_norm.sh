#!/bin/bash
#SBATCH --job-name=classify_vector_norm
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-56
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/classify_vector_norm-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/classify_vector_norm-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

# Batch par combo vectoriel (psd_*/psd_osc_*/aperiodic/higuchi_fd/perm_entropy/spec_entropy x 4 stades).
# --time provisoire, a corriger apres mesure reelle (jamais chronometre cette feature avant).

SAVE_ROOT=/home/alouis/scratch
BRANCH=${BRANCH:-2}

case $BRANCH in
    1) SAVE=dream_features_ica_1000hz_overlap          ;;
    2) SAVE=dream_features_noica_1000hz_overlap_normalized   ;;
    3) SAVE=dream_features_iclabel_1000hz_overlap  ;;
esac

KEYS=(psd_delta psd_theta psd_alpha psd_sigma psd_beta \
      psd_osc_delta psd_osc_theta psd_osc_alpha psd_osc_sigma psd_osc_beta \
      aperiodic higuchi_fd perm_entropy spec_entropy)
STATES=(S2 SWS NREM REM)

KEY_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
STATE_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))
KEY=${KEYS[$KEY_IDX]}
STATE=${STATES[$STATE_IDX]}

echo "=== classify_vector branch=${SAVE} key=${KEY} state=${STATE} ==="
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

python classify.py \
    --save-path       ${SAVE_ROOT}/${SAVE} \
    --n-jobs          $SLURM_CPUS_PER_TASK \
    --n-perm          1000 \
    --n-bootstraps    1000 \
    --checkpoint-every 50 \
    --key             ${KEY} \
    --state           ${STATE} \
    --skip-check \
    --normalize

echo "End: $(date)"
