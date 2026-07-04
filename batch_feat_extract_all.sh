#!/bin/bash
#SBATCH --job-name=feat_extract_all
#SBATCH --account=def-kjerbi
#SBATCH --array=1-3
#SBATCH --time=1:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/feat_extract_branch-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/feat_extract_branch-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

# Array :
#   1 -> preprocessed-ica     -> dream_features
#   2 -> preprocessed-noica   -> dream_features_noica
#   3 -> preprocessed-iclabel -> dream_features_iclabel
#
# Note --overwrite : ne pas ajouter en production. feat_extract skippe
# automatiquement les .npz existants (cache atomique par sujet/stade).
# N'utiliser --overwrite que si les .npz sont obsoletes (ex: changement
# de reference EEG). Dans ce cas, renommer d'abord les anciens dossiers
# en _CAR (archive), puis relancer sans --overwrite.

DERIV_ROOT=/home/alouis/scratch/dream_bids/derivatives_1000hz
SAVE_ROOT=/home/alouis/scratch

case $SLURM_ARRAY_TASK_ID in
    1)
        BRANCH=preprocessed-ica
        SAVE=dream_features_1000hz
        ;;
    2)
        BRANCH=preprocessed-noica
        SAVE=dream_features_noica_1000hz
        ;;
    3)
        BRANCH=preprocessed-iclabel
        SAVE=dream_features_iclabe_1000hz
        ;;
esac

echo "=== feat_extract branch=${BRANCH} save=${SAVE} ==="
echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Start: $(date)"

mkdir -p ${SAVE_ROOT}/${SAVE}
cd /home/alouis/dream-recall-alex

/home/alouis/mne_env/bin/python feat_extract_umap_fooof_v4.py \
    --deriv-path ${DERIV_ROOT}/${BRANCH} \
    --save-path  ${SAVE_ROOT}/${SAVE} \
    --n-jobs     $SLURM_CPUS_PER_TASK

echo "End: $(date)"
