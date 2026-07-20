#!/bin/bash
#SBATCH --job-name=recompute_fig3
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/alouis/logs_dream/recompute_fig3_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/recompute_fig3_%j.err

# Recompute des donnees de la Fig. 3 (chap.1 Arthur), en S2 :
#   1. panneau PSD (spectre Welch continu HR vs LR)  -> recompute_psd_spectrum_fig3.py
#   2. panneau T-values (pseudo-t two-sided maxstat)  -> recompute_ttest_fig3.py
# Le panneau LDA (droite) vient deja des classif existants + plot_topomap_psd_arthur.
#
# set -euo pipefail OBLIGATOIRE : sans lui, SLURM reporte COMPLETED meme si Python
# crash (masque les tracebacks). Cf mail node silent-failure.
set -euo pipefail

module load python/3.11 2>/dev/null || true
# venv Fir (meme que batch_preprocess_ica_only.sh, pattern eprouve)
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH

export PYTHONUNBUFFERED=1
# OMP_NUM_THREADS=1 : le parallelisme est au niveau joblib (--n-jobs), on evite la
# sur-souscription BLAS (cf note EPYC : threads BLAS x joblib = contention).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

STATE=S2
SAVE_PATH=/scratch/alouis/dream_features_noica_1000hz
DERIV_PATH=/scratch/alouis/dream_bids/derivatives/preprocessed-noica
OUT_DIR=/scratch/alouis/dream_features_noica_1000hz_corrected

echo "=== 1/2 : PSD spectrum (panneau gauche) ==="
python recompute_psd_spectrum_fig3.py \
    --deriv-path "$DERIV_PATH" \
    --out-dir    "$OUT_DIR" \
    --state      "$STATE" \
    --n-jobs     "$SLURM_CPUS_PER_TASK"

echo "=== 2/2 : T-values maxstat (panneau milieu) ==="
# --maxstat-scope electrodes = code Arthur (max sur 19 elec par bande).
# Ajouter --maxstat-scope both pour la variante texte-these (pool elec x bandes).
python recompute_ttest_fig3.py \
    --save-path "$SAVE_PATH" \
    --out-dir   "$OUT_DIR" \
    --state     "$STATE" \
    --n-perm    10000 \
    --maxstat-scope electrodes \
    --n-jobs    "$SLURM_CPUS_PER_TASK"

echo "=== FIG.3 recompute termine ==="
echo "Sorties dans : $OUT_DIR"
ls -lh "$OUT_DIR"/fig3_*_"$STATE".npz