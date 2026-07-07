#!/bin/bash
#SBATCH --job-name=efs_survivors
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/alouis/logs_dream/efs_%A.out
#SBATCH --error=/scratch/alouis/logs_dream/efs_%A.err

# ─────────────────────────────────────────────────────────────────────────────
# EFS cible sur les features survivantes (extension du chapitre 1, §1.3.3).
#
# Analyse EXPLORATOIRE : les features testees ont ete pre-selectionnees parce
# qu'elles survivent deja au test (double-dipping). Les p-values EFS sont
# optimistes et ne constituent pas un test d'hypothese independant. Le script
# ecrit cet avertissement en tete du CSV de sortie.
#
# Structure separee (comme classify.py, fidele a Arthur) : bootstraps stabilisent
# l'accuracy, permutations subject-level independantes forment la distribution
# nulle. Cache de projection tangent space intra-run_efs (projete 1x par appel,
# reutilise pour tous les combos).
#
# Cout estime : projection tangent ~9min x 399 appels (200 boot + 199 perm) par
# etat = ~60h sequentiel/etat. Sur 32 coeurs (~70% scaling effectif d'apres bench
# recent EPYC 9655) = ~2.7h/etat. S2 et SWS ont 3 survivantes chacun -> tient
# largement dans la limite 12h.
#
# Prerequis : results/pvalue_summary_table.csv doit exister (lance
# build_pvalue_summary_table.py avant). Le repertoire de features par defaut est
# le 1000Hz overlap actif.
#
# Usage :
#   sbatch --exclude=fc30555 batch_classify_efs.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module load python/3.11 2>/dev/null || true
source /home/alouis/mne_env/bin/activate

export PYTHONUNBUFFERED=1

SAVE_PATH="/scratch/alouis/dream_features_noica_1000hz_overlap"
CODE_DIR="/home/alouis/dream-recall-alex"

cd "$CODE_DIR"

echo "=== EFS survivors : demarrage $(date) ==="
echo "SAVE_PATH = $SAVE_PATH"
echo "CPUs      = $SLURM_CPUS_PER_TASK"

python -u classify_efs.py \
    --save-path    "$SAVE_PATH" \
    --n-jobs       "$SLURM_CPUS_PER_TASK" \
    --n-perm       199 \
    --n-bootstraps 200 \
    --alpha        0.05 \
    --pval-col     p_non_corrige_subject \
    --max-features 3 \
    --mode         both \
    --overwrite

echo "=== EFS survivors : fin $(date) ==="
