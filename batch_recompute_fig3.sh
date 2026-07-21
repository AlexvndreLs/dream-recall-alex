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
# Dossier DEDIE au recompute Fig.3 : on n'ecrit PAS directement dans _corrected/ (qui
# contient deja 32 fichiers maxstat valides) pour ne rien melanger ni risquer d'ecraser.
OUT_DIR=/scratch/alouis/dream_features_noica_1000hz_corrected/fig3_recompute
mkdir -p "$OUT_DIR"

echo "=== 1/2 : PSD spectrum (panneau gauche) ==="
python recompute_psd_spectrum_fig3.py \
    --deriv-path "$DERIV_PATH" \
    --out-dir    "$OUT_DIR" \
    --state      "$STATE" \
    --n-jobs     "$SLURM_CPUS_PER_TASK"

echo "=== 2/2 : T-values maxstat (panneau milieu) ==="
# REPLIQUE ARTHUR (FFX) : --level epoch (toutes epochs empilees, permutation niveau
# epoch). C'est ce que fait ttest.py d'Arthur. La version RFX correcte (--level
# subject) donne ~0 electrode sig en S2 : lancer separement si on veut le contraste.
# --zscore none = PSD brute (aucun z-score dans le code public d'Arthur ; equivalent
# au z-score global pour le t de Welch).
# --maxstat-scope electrodes = code Arthur (max sur 19 elec par bande).
# --drop-subjects vide par defaut ; passer 10 pour coller a Arthur (artefact FC2).
python recompute_ttest_fig3.py \
    --save-path "$SAVE_PATH" \
    --out-dir   "$OUT_DIR" \
    --state     "$STATE" \
    --n-perm    9999 \
    --level     epoch \
    --zscore    none \
    --maxstat-scope electrodes \
    --n-jobs    "$SLURM_CPUS_PER_TASK"

echo "=== FIG.3 recompute termine ==="
echo "Sorties dans : $OUT_DIR"
ls -lh "$OUT_DIR"/fig3_*_"$STATE".npz
