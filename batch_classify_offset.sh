#!/bin/bash
#SBATCH --job-name=classify_offset
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-4
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/classify_offset-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/classify_offset-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# LDA par electrode sur l'offset aperiodique, branche overlap, quatre etats.
# Calque sur batch_classify_vector.sh : memes n-perm, n-bootstraps,
# checkpoint-every, memes ressources. C'est la seule facon d'obtenir une
# colonne LDA comparable a celle de l'exposant, qui a ete produite par ce
# batch-la.
#
# aperiodic_offset n'est pas dans FEATURE_KEYS de config_v3, donc le controle
# de coherence de compute_global_n_trials ne le verifierait pas de toute
# facon. --skip-check est conserve par alignement sur batch_classify_vector.sh,
# et n'a aucun effet sur n_trials, qui reste calcule sur cov (61 epochs).

SAVE_ROOT=/home/alouis/scratch
SAVE=dream_features_noica_1000hz_overlap

KEY=aperiodic_offset
STATES=(S2 SWS NREM REM)
STATE=${STATES[$(( SLURM_ARRAY_TASK_ID - 1 ))]}

echo "=== classify_offset branch=${SAVE} key=${KEY} state=${STATE} ==="
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
    --skip-check

echo "End: $(date)"