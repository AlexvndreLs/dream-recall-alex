#!/bin/bash
set -euo pipefail
cd ~/dream-recall-alex
source ~/mne_env/bin/activate

# Chemins
EPOCH=/scratch/alouis/dream_features_noica_1000hz              # réplication Arthur
OVER=/scratch/alouis/dream_features_noica_1000hz_overlap       # figures propres
CORR=/scratch/alouis/dream_features_noica_1000hz_overlap_corrected
COORD=coord_cart_new.txt
P_ARTHUR=~/dream-recall-alex/plot          # sorties réplication
P_CLEAN=~/dream-recall-alex/plot_overlap   # sorties propres

echo "########## RÉPLICATION ARTHUR (epoch) ##########"

python plot_barplot_riemann_arthur.py \
  --save-path $EPOCH --out-dir $P_ARTHUR \
  --alpha 0.001 --perm-scheme epoch

python plot_topomap_psd_arthur.py \
  --save-path $EPOCH --out-dir $P_ARTHUR \
  --feature-family psd --alpha 0.001 --perm-scheme epoch \
  --correction maxstat --vmin 50 --vmax 60 --coord-file $COORD

echo "########## FIGURES PROPRES (subject, pooled) ##########"

python plot_barplot_riemann_clean.py \
  --save-path $OVER --corrected-path $CORR --out-dir $P_CLEAN --alpha 0.05

python plot_topomap_clean.py \
  --save-path $OVER --corrected-path $CORR --out-dir $P_CLEAN \
  --family psd_classic --coord-file $COORD

python plot_perm_null.py \
  --save-path $OVER --corrected-path $CORR --out-dir $P_CLEAN --family matrix

python plot_perm_null.py \
  --save-path $OVER --corrected-path $CORR --out-dir $P_CLEAN \
  --mode zoom --features cosp_sigma/S2 cosp_alpha/S2 cosp_delta/SWS cosp_delta/NREM

python plot_freq_profile.py \
  --save-path $OVER --corrected-path $CORR --out-dir $P_CLEAN

python plot_bootstrap_convergence.py \
  --save-path $OVER --out-dir $P_CLEAN \
  --features cosp_sigma/S2 cosp_delta/SWS cov/REM psd_sigma/S2

python plot_bootstrap_dispersion.py \
  --save-path $OVER --out-dir $P_CLEAN --family matrix

echo "########## TERMINÉ ##########"
ls -la $P_ARTHUR/*.png $P_CLEAN/*.png
