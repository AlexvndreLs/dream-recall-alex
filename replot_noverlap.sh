#!/bin/bash
#SBATCH --job-name=replot_nov
#SBATCH --account=rrg-kjerbi
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/replot_nov_%j.out
#
# Regenere sur la BRANCHE SANS RECOUVREMENT les trois figures :
#   barplot_riemann_subject_pooled_p0.05.png
#   barplot_psd_arthur_arthur_p0.05.png
#   barplot_psd_arthur_pooled_p0.05.png
# plus barplot_riemann_subject_raw_p0.05.png et barplot_psd_arthur_raw_p0.05.png,
# que les scripts produisent dans la meme passe.
#
# Se lance par sbatch ou directement en bash sur le noeud de connexion, les
# lignes #SBATCH etant alors de simples commentaires.

set -euo pipefail

cd ~/dream-recall-alex
source /home/alouis/mne_env/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

SRC=/scratch/alouis/dream_features_noica_1000hz
CORR=/scratch/alouis/dream_features_noica_1000hz_corrected
OUT=~/dream-recall-alex/plot
FIGS=~/dream-recall-alex/final_plotted_figures
STAMP=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------------------
# 0. Verification des prerequis. On s'arrete avant de tracer si un .npz manque,
#    plutot que de produire une figure trouee sans le dire.
# ---------------------------------------------------------------------------
echo "### verification des entrees"
manque=0
for st in S2 SWS NREM REM; do
  for k in cov cosp_delta cosp_theta cosp_alpha cosp_sigma cosp_beta \
           psd_delta psd_theta psd_alpha psd_sigma psd_beta; do
    [ -f "$SRC/results/${k}_${st}.npz" ] || { echo "  absent : results/${k}_${st}.npz"; manque=1; }
  done
  [ -f "$CORR/matrix_${st}_maxstat.npz" ]      || { echo "  absent : matrix_${st}_maxstat.npz"; manque=1; }
  [ -f "$CORR/psd_classic_${st}_maxstat.npz" ] || { echo "  absent : psd_classic_${st}_maxstat.npz"; manque=1; }
  for b in delta theta alpha sigma beta; do
    [ -f "$CORR/psd_${b}_${st}_maxstat_arthur.npz" ] \
      || { echo "  absent : psd_${b}_${st}_maxstat_arthur.npz"; manque=1; }
  done
done
if [ "$manque" -ne 0 ]; then
  echo "### des entrees manquent, rien n'a ete trace"
  exit 1
fi
echo "  toutes les entrees sont la"

# ---------------------------------------------------------------------------
# 1. Descripteurs matriciels, RFX, raw et pooled.
#    Pas de niveau maxstat electrodes ici : une matrice donne un test unique
#    par etat, il n'y a pas de dimension electrode a corriger.
# ---------------------------------------------------------------------------
echo "### matrices RFX, sans recouvrement"
python plot_barplot_riemann_clean.py \
  --save-path "$SRC" --corrected-path "$CORR" --out-dir "$OUT" --alpha 0.05

# ---------------------------------------------------------------------------
# 2. Puissances spectrales, RFX, raw, maxstat electrodes et pooled.
#    --pool-family psd_classic : sur cette branche la famille poolee porte ce
#    nom, le defaut psd du script correspond a la branche avec recouvrement.
# ---------------------------------------------------------------------------
echo "### psd RFX, sans recouvrement"
python plot_barplot_psd_arthur_clean.py \
  --save-path "$SRC" --corrected-path "$CORR" --out-dir "$OUT" \
  --alpha 0.05 --pool-family psd_classic

# ---------------------------------------------------------------------------
# 3. Copie vers le dossier des figures du rapport, avec sauvegarde de ce qui
#    s'y trouvait deja, les fichiers en place pouvant venir de la branche avec
#    recouvrement.
# ---------------------------------------------------------------------------
echo "### copie vers final_plotted_figures"
mkdir -p "$FIGS" "$FIGS/_remplacees_$STAMP"
for f in barplot_riemann_subject_raw_p0.05.png \
         barplot_riemann_subject_pooled_p0.05.png \
         barplot_psd_arthur_raw_p0.05.png \
         barplot_psd_arthur_arthur_p0.05.png \
         barplot_psd_arthur_pooled_p0.05.png ; do
  if [ -f "$FIGS/$f" ]; then
    mv "$FIGS/$f" "$FIGS/_remplacees_$STAMP/$f"
    echo "  ancienne version mise de cote : $f"
  fi
  cp "$OUT/$f" "$FIGS/$f"
done
rmdir "$FIGS/_remplacees_$STAMP" 2>/dev/null || true

echo "### termine"
ls -la "$FIGS" | grep -E "barplot_(riemann_subject|psd_arthur)"
