#!/bin/bash
#SBATCH --job-name=classify_dream
#SBATCH --account=def-kjerbi
#SBATCH --array=1-3
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/classify_branch-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/classify_branch-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

# Array :
#   1 -> dream_features     (branche ica, résultat principal)
#   2 -> dream_features_noica
#   3 -> dream_features_iclabel

SAVE_ROOT=/home/alouis/scratch

case $SLURM_ARRAY_TASK_ID in
    1) SAVE=dream_features       ;;
    2) SAVE=dream_features_noica ;;
    3) SAVE=dream_features_iclabel ;;
esac

echo "=== classify branch=${SAVE} ==="
echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Start: $(date)"

cd /home/alouis/dream-recall-alex
/home/alouis/mne_env/bin/python classify.py \
    --save-path  ${SAVE_ROOT}/${SAVE} \
    --n-jobs     $SLURM_CPUS_PER_TASK \
    --n-perm     1000 \
    --n-bootstraps 1000 \
    --skip-check

echo "End: $(date)"