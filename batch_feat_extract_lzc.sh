#!/bin/bash
#SBATCH --job-name=feat_extract_lzc
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-2
#SBATCH --time=10:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/feat_extract_lzc-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/feat_extract_lzc-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Deux bras, branche no-overlap noica 1000 Hz.
#
#   1 -> lzc      passe-bande 1-45 Hz, 1000 Hz conserve, PAS de decimation.
#                 Borne haute 45 Hz = FOOOF_FREQ_RANGE, donc meme bande que
#                 spec_entropy et que l'exposant aperiodique contre lequel le
#                 test de redondance la confronte. Filtrer avant est ce que
#                 fait toute la litterature (Tholke 2025 : 0.5-32 Hz ;
#                 Aamodt 2022 et Hohn 2024 : 250 Hz avec passe-haut).
#                 C'est CE bras qui alimente la ligne lzc de tab:complexite.
#
#   2 -> lzc_raw  signal brut, sans aucun filtre. Bras de CONTROLE, pas la
#                 ligne du rapport. Sert a CHIFFRER l'ecart entre les deux
#                 definitions plutot qu'a l'affirmer, et a rester comparable a
#                 higuchi_fd et perm_entropy, elles aussi calculees sur le
#                 signal large bande.
#
# Pourquoi le filtrage et non la decimation. Sur un 1/f borne 1-45 Hz auquel on
# ajoute du bruit 100-400 Hz valant 5 pourcent de la variance totale :
#     lzc  0.1403 -> 0.3574, et 0.1343 apres passe-bas 45 Hz
# Cinq pourcent de variance hors bande deplacent la mesure de 155 pourcent, et
# le passe-bas la restaure. La decimation, elle, ne change aucune accuracy :
# n est identique pour toutes les epoques, donc la normalisation n/log2(n) est
# une constante multiplicative, et la LDA a une dimension par electrode est
# invariante par changement d'echelle. Elle n'aurait fait qu'accelerer le
# calcul, ce n'est pas un argument de methode. On reste donc a 1000 Hz.
#
# Cout mesure du comptage LZ76 sur 30000 echantillons, par canal :
#     0.062 s sur un 1/f borne 1-45 Hz      (bras 1)
#     0.27  s sur du bruit blanc, pire cas  (borne haute du bras 2)
# Le bras 2 est donc plus lent, d'un facteur qui depend du contenu haute
# frequence reel et n'a pas ete mesure sur ces donnees. --time=10:00:00 est
# une borne haute posee sans mesure : lancer d'abord le test incremental
# (en bas de ce fichier) pour la resserrer.
#
# Memoire, MESUREE et non estimee. Le test sur sub-01 (650 epoques, 5h25 de
# sommeil score) a consomme 10.97 Go pour UN sujet seul. sub-24 fait 11h09
# d'enregistrement, c'est le plus gros de la cohorte, celui qui avait deja
# plante en OOM a 64G au preprocessing et qu'il avait fallu relancer a 128G.
# Il consommera plutot 15 a 18 Go.
#
# A 96G et 8 workers cela fait 12 Go par worker : si sub-24 tombe dans le meme
# lot que d'autres nuits longues, ca deborde. D'ou --n-jobs 6 plutot que
# SLURM_CPUS_PER_TASK, soit 16 Go par worker. Le cout est faible, environ
# 1.5 h au lieu de 1.1 h sur le bras filtre, contre un OOM en milieu de job.
#
# La cause de fond est corrigee dans le .py : le filtrage se fait desormais
# epoque par epoque et non sur le stade entier, ce qui evitait a filtfilt
# d'allouer plusieurs Go inutiles sur les nuits longues. L'equivalence des
# valeurs a ete verifiee bit a bit. --n-jobs 6 est la ceinture en plus des
# bretelles, le temps de voir un seff sur le job complet.
#
# --mem=96G / --cpus-per-task=8 reste le gabarit de batch_feat_extract_all.sh,
# qui est le ratio memoire/coeur eprouve sur ce cluster.
#
# Le cache est par (sujet, stade) : un timeout ne perd que le stade en cours,
# relancer le meme script reprend ou il s'etait arrete.

DERIV=/scratch/alouis/dream_bids/derivatives_1000hz/preprocessed-noica
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

case $SLURM_ARRAY_TASK_ID in
    1) KEY=lzc     ; EXTRA=""            ;;
    2) KEY=lzc_raw ; EXTRA="--no-filter" ;;
    *) echo "index d'array inattendu"; exit 1 ;;
esac

echo "=== feat_extract_lzc key=${KEY} branche=${BRANCH} save=${SAVE} ==="
echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Start: $(date)"

cd /home/alouis/dream-recall-alex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
source /home/alouis/mne_env/bin/activate

# NJOBS=8 sbatch ... pour repasser a 8 workers une fois le seff verifie.
NJOBS=${NJOBS:-6}

python feat_extract_lzc.py \
    --deriv-path "${DERIV}" \
    --save-path  "${SAVE}" \
    --key-name   "${KEY}" \
    ${EXTRA} \
    --n-jobs     "${NJOBS}"

echo "End: $(date)"

# --- test incremental a lancer AVANT celui-ci -------------------------------
# Un sujet, un coeur, sur un noeud interactif. Donne le temps par epoque et le
# nombre total d'epoques, donc de quoi calibrer --time.
#
#   salloc --account=rrg-kjerbi --time=1:00:00 --mem=16G --cpus-per-task=1 \
#          --exclude=fc30555
#   source /home/alouis/mne_env/bin/activate
#   cd /home/alouis/dream-recall-alex
#   python -c "import numba, antropy; print(numba.__version__, antropy.__version__)"
#   python feat_extract_lzc.py \
#       --deriv-path /scratch/alouis/dream_bids/derivatives_1000hz/preprocessed-noica \
#       --save-path  /scratch/alouis/dream_features_noica_1000hz \
#       --subjects   01 --n-jobs 1
#
# numba doit repondre. Sans lui antropy retombe sur du Python pur et le
# comptage LZ76 devient inutilisable, plusieurs ordres de grandeur.
#
# --- deux variantes optionnelles --------------------------------------------
# Tholke stricte, pour une comparaison exacte a l'etude de reference :
#   --highpass 0.5 --lowpass 32 --key-name lzc_tholke
# Decimation, uniquement si le temps de calcul devient un probleme. Sans effet
# sur les accuracies, verifier la coherence avec --lowpass (garde-fou Nyquist
# integre au script) :
#   --decimate 4 --key-name lzc_250hz
# Aamodt / Hohn, binarisation de l'enveloppe de Hilbert plutot que du signal :
#   --hilbert --key-name lzc_env
#
# --- branche overlap ---------------------------------------------------------
# La LZC ne passe pas par Welch : elle est donc IDENTIQUE entre la branche
# overlap et la branche no-overlap, la segmentation en epoques etant la meme
# (i += 30 dans les deux). Inutile de la recalculer, des liens durs suffisent :
#   cd /scratch/alouis
#   for k in lzc lzc_raw; do
#     mkdir -p dream_features_noica_1000hz_overlap/$k
#     ln dream_features_noica_1000hz/$k/*.npz \
#        dream_features_noica_1000hz_overlap/$k/
#   done
