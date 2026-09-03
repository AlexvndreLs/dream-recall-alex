#!/bin/bash
# Post-traitement de la LZC, apres batch_classify_lzc.sh.
# Quelques minutes, a lancer sur un noeud de connexion, pas besoin de SLURM.
#
#   bash postprocess_lzc.sh
#   KEY=lzc_raw bash postprocess_lzc.sh     # pour le bras de controle

set -euo pipefail

# Branche : BRANCH=overlap (defaut) ou BRANCH=no_overlap.
# L'overlap designe le recouvrement des fenetres de Welch dans l'estimation du
# spectre. La LZC ne passe pas par Welch, elle est comptee sur le signal
# temporel, et la segmentation en epoques est la meme des deux cotes : les
# fichiers lzc/ sont donc IDENTIQUES entre branches, des liens durs suffisent
# (voir bas de batch_feat_extract_lzc.sh). Ce qui change ce sont les accuracies
# et les p, parce que n_trials vient de cov et que la comparaison de redondance
# se fait contre aperiodic, qui lui depend de Welch.
BRANCH=${BRANCH:-overlap}
case "${BRANCH}" in
    overlap)    SAVE=/scratch/alouis/dream_features_noica_1000hz_overlap ;;
    no_overlap) SAVE=/scratch/alouis/dream_features_noica_1000hz ;;
    *) echo "BRANCH doit valoir overlap ou no_overlap, recu : ${BRANCH}"; exit 1 ;;
esac
CORR="${SAVE}_corrected"
OUTDIR=/home/alouis/dream-recall-alex/plot_lzc_${BRANCH}
KEY=${KEY:-lzc}
REF=${REF:-aperiodic}

cd /home/alouis/dream-recall-alex
source /home/alouis/mne_env/bin/activate
mkdir -p "${OUTDIR}"

echo "=== 1. verification des 4 resultats ==="
for st in S2 SWS NREM REM; do
    f="${SAVE}/results/${KEY}_${st}.npz"
    [ -f "$f" ] && echo "  OK        $f" || echo "  MANQUANT  $f"
done

echo "=== 2. correction max-stat sur les 19 electrodes (mode arthur) ==="
# La LZC est une mesure isolee : pas de famille de bandes, donc la correction
# pooled coincide avec la correction par electrode. Meme traitement que
# higuchi_fd, perm_entropy et spec_entropy, et c'est exactement ce que dit la
# section 3 du rapport : "la correction en commun coincide donc avec la
# correction par electrode".
python compute_maxstat_correction.py \
    --save-path   "${SAVE}" \
    --output-path "${CORR}" \
    --family-name unused \
    --mode        arthur \
    --keys        "${KEY}"
# Produit ${CORR}/${KEY}_{S2,SWS,NREM,REM}_maxstat_arthur.npz, avec ch_names,
# real_values, pvals_corrected, null_max, n_perm, n_tests.

echo "=== 3. tableau de synthese ==="
# build_pvalue_summary_table.py decouvre les cles en listant results/*.npz.
# lzc n'etant ni dans MATRIX_FAMILY ni dans PSD_CLASSIC_FAMILY, il tombe dans
# la branche isolee ou pooled == arthur. Aucun patch necessaire.
python build_pvalue_summary_table.py --save-path "${SAVE}"

echo "=== 4. barplot ==="
# Necessite patch_barplot_add_lzc.py applique au prealable.
python plot_barplot_vector_clean.py \
    --save-path      "${SAVE}" \
    --corrected-path "${CORR}" \
    --out-dir        "${OUTDIR}" \
    --alpha 0.05

echo "=== 5. redondance, etape 1 (Spearman contre ${REF}) ==="
# Quelques secondes, pas de classification. L'etape 2 est un job SLURM separe
# (batch_lzc_redundancy.sh) parce qu'elle coute deux classifications
# vectorielles completes.
python test_lzc_redundancy.py \
    --save-path      "${SAVE}" \
    --out-dir        "${OUTDIR}" \
    --key-complexity "${KEY}" \
    --key-reference  "${REF}" \
    --step 1

echo
echo "Fait. Pour remplir tab:complexite, lire dans"
echo "  ${SAVE}/results/pvalue_summary_table.csv   lignes feature=${KEY}"
echo "les colonnes best_electrode, accuracy_pct, p_non_corrige_subject,"
echo "p_maxstat_arthur_subject."
echo
echo "Etape suivante : sbatch batch_lzc_redundancy.sh"
