#!/bin/bash
# t-tests RFX sur toutes les mesures de la figure format Tholke, branche overlap.
#
# Leger : chaque couple key x state est un ttest_ind sur (36, 19) repete 9999
# fois. Se lance sur un noeud de connexion, pas besoin de SLURM. Compter
# quelques minutes au total, dominees par la lecture des .npz atomiques.
#
# Les quatre definitions de puissance oscillatoire vivent dans trois racines
# differentes : brute et ratio dans la branche overlap principale, sub et
# logsub dans leurs propres branches. Ces deux dernieres portent un nom sans
# "_overlap" mais ont bien ete extraites avec OVERLAP = 500, leurs batchs
# pointant --cov-source vers la branche overlap.
#
# aperiodic_offset n'est traite que si l'extraction est passee.

set -euo pipefail
export PYTHONUNBUFFERED=1
source /home/alouis/mne_env/bin/activate
cd /home/alouis/dream-recall-alex

SCR=/scratch/alouis
NPERM=9999

# --- exposant, et offset si disponible, branche overlap ----------------------
OFFSET_KEY=""
if [ -d "${SCR}/dream_features_noica_1000hz_overlap/aperiodic_offset" ]; then
    OFFSET_KEY="aperiodic_offset"
    echo "offset detecte, inclus dans le lot."
else
    echo "offset absent, traite plus tard. Relancer ce script apres extraction."
fi

python ttest_vector_rfx.py \
    --save-path ${SCR}/dream_features_noica_1000hz_overlap \
    --out-dir   ${SCR}/dream_features_noica_1000hz_overlap_ttest \
    --keys aperiodic ${OFFSET_KEY} \
    --n-perm ${NPERM}

# --- definition brute et definition ratio, branche overlap -------------------
python ttest_vector_rfx.py \
    --save-path ${SCR}/dream_features_noica_1000hz_overlap \
    --out-dir   ${SCR}/dream_features_noica_1000hz_overlap_ttest \
    --keys psd_delta psd_theta psd_alpha psd_sigma psd_beta \
           psd_osc_delta psd_osc_theta psd_osc_alpha psd_osc_sigma psd_osc_beta \
    --n-perm ${NPERM}

# --- definition sub ----------------------------------------------------------
python ttest_vector_rfx.py \
    --save-path ${SCR}/dream_features_noica_1000hz_sub \
    --out-dir   ${SCR}/dream_features_noica_1000hz_sub_ttest \
    --keys psd_sub_delta psd_sub_theta psd_sub_alpha psd_sub_sigma psd_sub_beta \
    --n-perm ${NPERM}

# --- definition logsub -------------------------------------------------------
python ttest_vector_rfx.py \
    --save-path ${SCR}/dream_features_noica_1000hz_logsub \
    --out-dir   ${SCR}/dream_features_noica_1000hz_logsub_ttest \
    --keys psd_logsub_delta psd_logsub_theta psd_logsub_alpha \
           psd_logsub_sigma psd_logsub_beta \
    --n-perm ${NPERM}

echo
echo "Termine. Resultats dans *_ttest/ , un npz par key x state plus un CSV."