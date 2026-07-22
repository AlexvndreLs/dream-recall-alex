#!/bin/bash
#SBATCH --job-name=fig3_ttest_cmp
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/alouis/logs_dream/fig3_ttest_cmp_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/fig3_ttest_cmp_%j.err

# Compare la colonne t-values de la Fig.3 (chap.1 Arthur) en deux versions :
#   1. version correcte  (two-tailed symetrique : |t_obs| vs |t_perm|)
#   2. replique exacte Arthur (--arthur-pval-bug : biais de signe E6, effets HR<LR
#      invisibles), pour la section "reproduction" de la presentation.
# Puis genere les deux topomaps dans dream-recall-alex/plot/.
#
# set -euo pipefail OBLIGATOIRE : sans lui, SLURM reporte COMPLETED meme si Python
# crash (masque les tracebacks). Cf mail node silent-failure.
set -euo pipefail

module load python/3.11 2>/dev/null || true
# venv Fir (meme que batch_recompute_fig3.sh, pattern eprouve)
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /home/alouis/dream-recall-alex

SAVE=/scratch/alouis/dream_features_noica_1000hz
RESULTS=$SAVE/results
CORR=/scratch/alouis/dream_features_noica_1000hz_corrected
OUT_CORRECT=$CORR/fig3_recompute_correct
OUT_ARTHUR=$CORR/fig3_recompute_arthurbug
PLOT=/home/alouis/dream-recall-alex/plot
STATE=S2

mkdir -p "$OUT_CORRECT" "$OUT_ARTHUR" "$PLOT" /scratch/alouis/logs_dream

echo "=== 1/4 : recompute ttest, version correcte (two-tailed symetrique) ==="
python recompute_ttest_fig3.py \
    --save-path "$SAVE" \
    --out-dir   "$OUT_CORRECT" \
    --state "$STATE" --n-perm 9999 --level epoch --zscore none \
    --maxstat-scope electrodes --drop-subjects 10 \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --overwrite

echo "=== 2/4 : recompute ttest, replique exacte Arthur (biais de signe E6) ==="
python recompute_ttest_fig3.py \
    --save-path "$SAVE" \
    --out-dir   "$OUT_ARTHUR" \
    --state "$STATE" --n-perm 9999 --level epoch --zscore none \
    --maxstat-scope electrodes --drop-subjects 10 --arthur-pval-bug \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --overwrite

echo "=== 3/4 : figure version correcte -> plot/ ==="
python plot_fig3_arthur_topomaps.py \
    --save-path "$SAVE" \
    --in-dir    "$OUT_CORRECT" \
    --results   "$RESULTS" \
    --coord-file coord_cart_new.txt \
    --state "$STATE" --out "$PLOT/fig3_arthur_${STATE}_correct.png"

echo "=== 4/4 : figure replique exacte Arthur -> plot/ ==="
python plot_fig3_arthur_topomaps.py \
    --save-path "$SAVE" \
    --in-dir    "$OUT_ARTHUR" \
    --results   "$RESULTS" \
    --coord-file coord_cart_new.txt \
    --state "$STATE" --out "$PLOT/fig3_arthur_${STATE}_arthurbug.png"

echo "=== termine. Figures dans $PLOT ==="
ls -lh "$PLOT"/fig3_arthur_${STATE}_*.png