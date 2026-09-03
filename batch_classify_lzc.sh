#!/bin/bash
#SBATCH --job-name=classify_lzc
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-4
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/classify_lzc-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/classify_lzc-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Un job par etat, 4 au total. Decalque de batch_classify_vector.sh, memes
# ressources et memes hyperparametres (1000 bootstraps, 1000 permutations
# sujet), pour que la ligne lzc du tableau soit strictement comparable aux
# trois autres mesures de complexite.
#
# classify.py n'a besoin d'AUCUNE modification : is_matrix_feature("lzc")
# est faux, donc la route vectorielle s'applique (LDA par electrode), et
# load_atomic va chercher lzc/lzc_s{XX}_{stage}.npz de lui-meme.
#
# --skip-check est indispensable : la verification d'integrite parcourt
# FEATURE_KEYS de config_v3.py, ou lzc ne figure pas. C'est deja le cas dans
# batch_classify_vector.sh, aucun changement de comportement.
#
# n_trials est calcule depuis cov comme pour toutes les autres features, donc
# le nombre d'epoques par sujet est identique. Le seed de permutation est
# _seed('perm', state, ...), independant de la key : les perm_accs de lzc
# sont donc alignees avec celles des autres features du meme etat, et
# poolables si besoin.

# Branche : BRANCH=overlap (defaut) ou BRANCH=no_overlap.
# L'overlap designe le recouvrement des fenetres de Welch dans l'estimation du
# spectre. La LZC ne passe pas par Welch, elle est comptee sur le signal
# temporel, et la segmentation en epoques est la meme des deux cotes : les
# fichiers lzc/ sont donc IDENTIQUES entre branches, des liens durs suffisent
# (voir bas de batch_feat_extract_lzc.sh). Ce qui change ce sont les accuracies
# et les p, parce que n_trials vient de cov et que la comparaison de redondance
# se fait contre aperiodic, qui lui depend de Welch.
BRANCH=${BRANCH:-overlap}
case "${BRANCH}" in
    overlap)    SAVE=/scratch/alouis/dream_features_noica_1000hz_overlap ;;
    no_overlap) SAVE=/scratch/alouis/dream_features_noica_1000hz ;;
    *) echo "BRANCH doit valoir overlap ou no_overlap, recu : ${BRANCH}"; exit 1 ;;
esac
CORR="${SAVE}_corrected"
OUTDIR=/home/alouis/dream-recall-alex/plot_lzc_${BRANCH}
KEY=${KEY:-lzc}
STATES=(S2 SWS NREM REM)
STATE=${STATES[$(( SLURM_ARRAY_TASK_ID - 1 ))]}

echo "=== classify key=${KEY} state=${STATE} branche=${BRANCH} save=${SAVE} ==="
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
    --save-path        "${SAVE}" \
    --n-jobs           "${SLURM_CPUS_PER_TASK}" \
    --n-perm           1000 \
    --n-bootstraps     1000 \
    --checkpoint-every 50 \
    --key              "${KEY}" \
    --state            "${STATE}" \
    --skip-check

echo "End: $(date)"

# Pour la variante bande limitee : KEY=lzc_bl sbatch batch_classify_lzc.sh
