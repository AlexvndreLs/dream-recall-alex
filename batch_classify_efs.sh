#!/bin/bash
#SBATCH --job-name=efs_survivors
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
# Ressources : 16 workers (--cpus-per-task=16) + 128G. Le run precedent a 32
# workers/64G a fait OOM (Memory 99.99% de 64G, tue a 9min avant de reveler le
# vrai pic). Cause : prefer="processes" duplique le fold_cache (324 folds projetes)
# dans chaque worker -> 32 caches simultanes saturent la RAM. Fix rapide : 2x moins
# de workers (moins de caches simultanes) + marge memoire large. CPU efficiency
# etait 61% sur 32 coeurs (processes OK, le probleme etait purement memoire).
# Solution de fond non appliquee ici : streaming du cache fold-par-fold (diviserait
# le pic memoire par ~324, permettrait de revenir a 32 coeurs).
# Cout : ~3h/etat intra (S2, SWS) + ~1h cross = ~7h, un peu plus lent qu'a 32 workers.
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
    --pval-col     p_maxstat_pooled_subject \
    --max-features 3 \
    --mode         both \
    --overwrite

echo "=== EFS survivors : fin $(date) ==="
