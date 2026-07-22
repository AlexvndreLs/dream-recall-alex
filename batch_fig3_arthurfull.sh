#!/bin/bash
#SBATCH --job-name=fig3_arthurfull
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/alouis/logs_dream/fig3_arthurfull_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/fig3_arthurfull_%j.err

# Reproduction FIDELE de la Fig.3 d'Arthur (chap.1), ses DEUX erreurs incluses :
#   --zscore subject   : z-score par sujet (annule l'effet de groupe, cf prepare_data)
#   --arthur-pval-bug  : biais de signe two-tailed (effets HR<LR invisibles, E6)
# Resultat attendu : colonne t-values quasi vide, aucune etoile, comme sa figure.
# C'est la version "reproduction a l'identique" pour la presentation, PAS un resultat.
set -euo pipefail

module load python/3.11 2>/dev/null || true
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /home/alouis/dream-recall-alex

SAVE=/scratch/alouis/dream_features_noica_1000hz
RESULTS=$SAVE/results
OUT=/scratch/alouis/dream_features_noica_1000hz_corrected/fig3_recompute_arthurfull
FIG=/home/alouis/dream-recall-alex/figures
STATE=S2

mkdir -p "$OUT" "$FIG" /scratch/alouis/logs_dream

echo "=== 1/2 : recompute ttest, reproduction Arthur complet (zscore sujet + bug signe) ==="
python recompute_ttest_fig3.py \
    --save-path "$SAVE" \
    --out-dir   "$OUT" \
    --state "$STATE" --n-perm 9999 --level epoch --zscore subject \
    --maxstat-scope electrodes --drop-subjects 10 --arthur-pval-bug \
    --n-jobs "$SLURM_CPUS_PER_TASK"

echo "=== 2/2 : figure -> figures/ ==="
python plot_fig3_arthur_topomaps.py \
    --save-path "$SAVE" \
    --in-dir    "$OUT" \
    --results   "$RESULTS" \
    --coord-file coord_cart_new.txt \
    --state "$STATE" --out "$FIG/fig3_arthur_${STATE}_arthurfull.png"

echo "=== termine ==="
ls -lh "$FIG/fig3_arthur_${STATE}_arthurfull.png"