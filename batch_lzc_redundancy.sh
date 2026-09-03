#!/bin/bash
#SBATCH --job-name=lzc_redundancy
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-4
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/lzc_redundancy-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/lzc_redundancy-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Etape 2 du protocole de redondance, un job par etat.
#
# L'etape 1 (Spearman) tourne en quelques secondes et n'a pas besoin de SLURM,
# elle est appelee depuis postprocess_lzc.sh. Ce script ne fait que l'etape 2,
# qui coute deux classifications vectorielles completes : pour chaque
# electrode, une LDA sur l'exposant seul et une LDA sur exposant + LZC, sur le
# meme echantillon bootstrap et les memes splits.
#
# --n-bootstraps 200 et non 1000 : le gain est une difference APPARIEE, les
# deux modeles voyant exactement les memes donnees et les memes splits. Sa
# variance est donc bien plus faible que celle de chaque accuracy prise
# isolement, et 200 tirages suffisent. Passer a 1000 multiplie le temps par 5
# pour un intervalle a peine plus serre.
#
# --n-perm 0 : l'intervalle bootstrap suffit au critere du rapport, qui est
# descriptif ("un gain inferieur a 1 ou 2 points signifie que la complexite
# n'ajoute rien de decodable"). Mettre N_PERM=1000 dans l'environnement si on
# veut une valeur de p publiable sur le gain, mais le cout est alors multiplie
# par 6 et --time doit etre revu en consequence.

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
NBOOT=${NBOOT:-200}
NPERM=${NPERM:-0}

STATES=(S2 SWS NREM REM)
STATE=${STATES[$(( SLURM_ARRAY_TASK_ID - 1 ))]}

echo "=== redundancy step 2 : ${KEY} vs ${REF}, state=${STATE}, branche=${BRANCH} ==="
echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Start: $(date)"

cd /home/alouis/dream-recall-alex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export FLEXIBLAS_NUM_THREADS=1
source /home/alouis/mne_env/bin/activate

# Un dossier de sortie par etat : les 4 jobs ecrivent en parallele et
# ecriraient sinon le meme CSV. Concatener ensuite (commande en bas).
mkdir -p "${OUTDIR}/step2_${STATE}"

python test_lzc_redundancy.py \
    --save-path      "${SAVE}" \
    --out-dir        "${OUTDIR}/step2_${STATE}" \
    --key-complexity "${KEY}" \
    --key-reference  "${REF}" \
    --states         "${STATE}" \
    --step           2 \
    --n-bootstraps   "${NBOOT}" \
    --n-perm         "${NPERM}" \
    --n-jobs         "${SLURM_CPUS_PER_TASK}"

echo "End: $(date)"

# --- une fois les 4 jobs termines -------------------------------------------
# Concatener les 4 CSV et regenerer la figure des 4 etats ensemble :
#
#   cd /home/alouis/dream-recall-alex
#   source /home/alouis/mne_env/bin/activate
#   python - <<'PY'
#   from pathlib import Path
#   import pandas as pd
#   out = Path("plot_noverlap_lzc")
#   fs = sorted(out.glob("step2_*/redundancy_step2_lzc_vs_aperiodic.csv"))
#   df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
#   order = {s: i for i, s in enumerate(["S2", "SWS", "NREM", "REM"])}
#   df = df.sort_values("state", key=lambda c: c.map(order))
#   df.to_csv(out / "redundancy_step2_lzc_vs_aperiodic.csv", index=False)
#   from test_lzc_redundancy import plot_step2
#   plot_step2(df, out, "lzc", "aperiodic")
#   PY
#
# Appliquer le meme protocole aux trois autres mesures de complexite, qui n'y
# ont jamais ete soumises non plus (le rapport le note comme piste ouverte) :
#   for k in higuchi_fd perm_entropy spec_entropy; do
#       KEY=$k sbatch batch_lzc_redundancy.sh
#   done
