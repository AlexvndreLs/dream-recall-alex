#!/bin/bash
#SBATCH --job-name=fig3_ffx_rfx_ovl
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/alouis/logs_dream/fig3_ffx_rfx_ovl_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/fig3_ffx_rfx_ovl_%j.err

# Planche 2x2 FFX vs RFX (t-values x accuracies), dataset OVERLAP.
# Les deux t-tests sont recalcules sur le meme dataset que les accuracies,
# pour ne pas melanger overlap et no-overlap dans une meme figure.
#
# set -euo pipefail OBLIGATOIRE : sans lui SLURM reporte COMPLETED meme si Python
# crash, ce qui masque les tracebacks.
set -euo pipefail

module load python/3.11 2>/dev/null || true
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /home/alouis/dream-recall-alex

SAVE=/scratch/alouis/dream_features_noica_1000hz_overlap
CORR=/scratch/alouis/dream_features_noica_1000hz_overlap_corrected
FIGS=/home/alouis/dream-recall-alex/figures
STATE=S2

mkdir -p "$CORR/fig3_rfx_correct" "$CORR/fig3_ffx_correct" "$FIGS" \
         /scratch/alouis/logs_dream

echo "=== 1/3 : t-test RFX (niveau sujet, conforme these) ==="
python recompute_ttest_fig3.py \
    --save-path "$SAVE" \
    --out-dir   "$CORR/fig3_rfx_correct" \
    --state "$STATE" --n-perm 1000 --level subject --zscore none \
    --maxstat-scope both --drop-subjects 10 \
    --n-jobs "$SLURM_CPUS_PER_TASK" --overwrite

echo "=== 2/3 : t-test FFX (niveau epoch, contrefactuel) ==="
python recompute_ttest_fig3.py \
    --save-path "$SAVE" \
    --out-dir   "$CORR/fig3_ffx_correct" \
    --state "$STATE" --n-perm 9999 --level epoch --zscore none \
    --maxstat-scope electrodes --drop-subjects 10 \
    --n-jobs "$SLURM_CPUS_PER_TASK" --overwrite

echo "=== 3/3 : planche 2x2 ==="
python plot_ffx_vs_rfx_2x2.py \
    --ttest-ffx "$CORR/fig3_ffx_correct" \
    --ttest-rfx "$CORR/fig3_rfx_correct" \
    --results   "$SAVE/results" \
    --coord-file coord_cart_new.txt \
    --state "$STATE" --alpha-ffx 0.001 --alpha-rfx 0.05 \
    --out "$FIGS/fig3_ffx_vs_rfx_${STATE}_overlap.png"

echo "=== termine ==="
ls -lh "$FIGS"/fig3_ffx_vs_rfx_${STATE}_overlap.png
