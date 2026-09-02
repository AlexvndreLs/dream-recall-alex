#!/bin/bash
#SBATCH --job-name=extract_offset
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-2
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/extract_offset_branch-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/extract_offset_branch-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

# Extraction de l'offset aperiodique FOOOF, absent de toutes les branches.
#
# Array :
#   1 -> dream_features_noica_1000hz          overlap Welch = 0   (replication stricte)
#   2 -> dream_features_noica_1000hz_overlap  overlap Welch = 500
#
# L'overlap n'est PAS lu depuis config_v3.py : la config contient aujourd'hui
# OVERLAP = 500, valeur des branches _overlap, alors que la branche de
# replication stricte a ete produite avec un overlap nul. Le passer en dur ici,
# par branche, et laisser --verify controler que c'etait le bon.
#
# --verify recalcule l'exposant et le compare a celui deja stocke sous la key
# "aperiodic". Lire la ligne VERIFICATION en fin de log avant de faire quoi que
# ce soit des offsets produits.

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

DERIV=/home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica

case $SLURM_ARRAY_TASK_ID in
    1)
        SAVE=/scratch/alouis/dream_features_noica_1000hz
        WOVERLAP=0
        ;;
    2)
        SAVE=/scratch/alouis/dream_features_noica_1000hz_overlap
        WOVERLAP=500
        ;;
    *)
        echo "ERREUR : index d'array ${SLURM_ARRAY_TASK_ID} inconnu, rien a faire." >&2
        exit 1
        ;;
esac

# Le dossier cible doit deja exister et contenir aperiodic/ : on ecrit DANS une
# branche existante, on n'en cree pas une nouvelle par erreur de frappe.
if [ ! -d "${SAVE}/aperiodic" ]; then
    echo "ERREUR : ${SAVE}/aperiodic introuvable. Mauvaise branche ?" >&2
    exit 1
fi

echo "=== extract offset save=${SAVE} welch_overlap=${WOVERLAP} ==="
echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Start: $(date)"

cd /home/alouis/dream-recall-alex

/home/alouis/mne_env/bin/python feat_extract_offset.py \
    --deriv-path    ${DERIV} \
    --save-path     ${SAVE} \
    --welch-overlap ${WOVERLAP} \
    --n-jobs        $SLURM_CPUS_PER_TASK \
    --verify

echo "End: $(date)"