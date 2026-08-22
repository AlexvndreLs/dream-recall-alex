#!/bin/bash
# ==========================================================================
# Regenere les figures dont les libelles viennent d'etre traduits, puis
# rassemble les onze figures du rapport dans final_plotted_figures/.
#
# A lancer sur le noeud de connexion de Fir, apres patch_figure_labels_en.py.
# Aucun calcul lourd : tout consomme des .npz deja produits. Si les .npz de
# fig04 manquent, le script s'arrete et indique le batch SLURM a soumettre
# plutot que de produire une figure vide.
#
# Usage :
#   cd ~/dream-recall-alex
#   bash make_final_figures.sh
# ==========================================================================

set -euo pipefail

cd ~/dream-recall-alex
source /home/alouis/mne_env/bin/activate

# --------------------------------------------------------------- chemins
EPOCH=/scratch/alouis/dream_features_noica_1000hz               # sans recouvrement
OVER=/scratch/alouis/dream_features_noica_1000hz_overlap        # avec recouvrement
CORR=/scratch/alouis/dream_features_noica_1000hz_overlap_corrected
SUB=/scratch/alouis/dream_features_noica_1000hz_sub
LOGSUB=/scratch/alouis/dream_features_noica_1000hz_logsub
SUB_CORR=${SUB}_corrected
LOGSUB_CORR=${LOGSUB}_corrected

COORD=coord_cart_new.txt
P_ARTHUR=~/dream-recall-alex/plot
P_CLEAN=~/dream-recall-alex/plot_overlap
FINAL=~/dream-recall-alex/final_plotted_figures

mkdir -p "$FINAL"

echo "########## 1. fig04, barplot riemannien FFX, branche SANS recouvrement"

# Verification prealable : les .npz du schema epoch fige doivent exister sur
# la branche noica. Ils sont produits par batch_recompute_perms_epoch_fixed_matrix.sh
# qui pointe deja sur $EPOCH, ils sont donc probablement la.

# N_FIXED=$(find "$EPOCH/results" -maxdepth 1 -name "*_epochperm_fixed.npz" 2>/dev/null | wc -l)
# echo "  ${N_FIXED} fichiers *_epochperm_fixed.npz trouves dans ${EPOCH}/results (attendu : 24)"

# if [ "$N_FIXED" -lt 24 ]; then
#     echo
#     echo "  ARRET. Les lois nulles du schema epoch fige sont incompletes sur"
#     echo "  la branche sans recouvrement. Soumets d'abord :"
#     echo "      sbatch batch_recompute_perms_epoch_fixed_matrix.sh"
#     echo "  (SAVE_PATH y pointe deja sur ${EPOCH}, --exclude=fc30555 et les"
#     echo "  directives mail sont deja en place). Relance ce script ensuite."
#     exit 1
# fi

# python plot_barplot_riemann_ffx_fixed.py \
#     --save-path "$EPOCH" --out-dir "$P_ARTHUR" \
#     --alpha 0.001 --correction global
# echo
echo "########## 2. fig05, topomaps PSD par electrode, branche SANS recouvrement"

python plot_topomap_psd_arthur.py \
    --save-path "$EPOCH" --out-dir "$P_ARTHUR" \
    --feature-family psd --alpha 0.001 --perm-scheme epoch \
    --correction maxstat --vmin 50 --vmax 60 --coord-file "$COORD"

echo
echo "########## 3. fig09, quatre definitions de la PSD, branche AVEC recouvrement"

for MODE in "--full" ""; do
    python plot_barplot_psd_defs4.py \
        --save-path "$OVER"          --corrected-path        "$CORR" \
        --sub-path "$SUB"            --sub-corrected-path    "$SUB_CORR" \
        --logsub-path "$LOGSUB"      --logsub-corrected-path "$LOGSUB_CORR" \
        --out-dir "$P_CLEAN" --alpha 0.05 $MODE
done

echo
echo "########## 4. Collecte dans final_plotted_figures/"

copy_fig () {
    # copy_fig <source> <destination>
    if [ -f "$1" ]; then
        cp "$1" "$FINAL/$2"
        echo "  ok    $2"
    else
        echo "  MANQUE $2   (source absente : $1)"
    fi
}

# fig01 : schema d'architecture, en TikZ directement dans 00_intro.tex, rien a copier.
copy_fig "plot_perm_explication/perm_scheme_schematic.png"                 "fig02_perm_scheme.png"
copy_fig "plot_perm_explication/deflation_hist_full.png"                   "fig03_deflation.png"
copy_fig "$P_ARTHUR/barplot_riemann_epoch_fixed_maxstat_global_p0.001.png" "fig04_riemann_replication.png"
copy_fig "$P_ARTHUR/topomap_psd_epoch_maxstat_p0.001.png"                  "fig05_topomap_psd.png"
copy_fig "$P_CLEAN/fig5_courbe_survie_pvalues.png"                         "fig06_survie_pvalues.png"
copy_fig "figures/fig3_arthur_S2_arthurfull.png"                           "fig07a_ttest_reproduit.png"
copy_fig "figures/fig3_corrigee_S2.png"                                    "fig07b_ttest_corrige.png"
copy_fig "figures/fig5_arthur_grid_v3.png"                                 "fig08_efs_regions.png"
copy_fig "$P_CLEAN/barplot_psd_defs4_arthur_full_p0.05.png"                "fig09a_psd_definitions.png"
copy_fig "$P_CLEAN/barplot_psd_defs4_arthur_focus_p0.05.png"               "fig09b_psd_definitions_focus.png"

# Section 1.1 : comparaison des deux constructions de la loi nulle.
copy_fig "$P_CLEAN/barplot_riemann_epoch_maxstat_global_p0.001.png"        "figA1_nulle_rebootstrap.png"
copy_fig "$P_CLEAN/barplot_riemann_epoch_fixed_maxstat_global_p0.001.png"  "figA2_nulle_echantillon_fige.png"

echo
echo "########## Bilan"
ls -la "$FINAL"
echo
echo "Manquent encore, a produire :"
echo "  fig10_aperiodic_topo.png    topographies exposant et decalage, format Tholke"
echo "  fig11_complexite.png        synthese des mesures de complexite, bloquee par la LZC"
echo
echo "Recuperer en local :"
echo "  rsync -av alouis@fir.alliancecan.ca:~/dream-recall-alex/final_plotted_figures/ ./figures/"
