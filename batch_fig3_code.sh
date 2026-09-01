#!/bin/bash
#SBATCH --job-name=fig3_code
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/alouis/logs_dream/fig3_code_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/fig3_code_%j.err

# Colonne 3 de la Fig.3, version CODE PUBLIE d'Arthur (ttest.py) :
#   --level epoch   : ttest.py l.34-35 concatene toutes les epochs. Le bloc de
#                     moyenne par sujet l.36-38 est commente, et il est place
#                     APRES le concatenate, donc sans effet meme decommente.
#   --n-perm        : 9999 chez lui (ttest.py l.14). Pilotable ici :
#                     N_PERM=1000 sbatch batch_fig3_code.sh  pour un premier
#                     passage et une mesure du temps. Attention, a 1000 perms la
#                     p-value est un multiple de 1/999 = 0.001001, donc p<0.001
#                     n'est atteignable que par p=0 exactement.
#   --maxstat-scope electrodes : un appel ttest_perm_unpaired par couple
#                     stade-bande, correction "maxstat" par defaut
#                     (ttest_perm_indep.py l.77), max sur les 19 electrodes.
#   --arthur-pval-bug : biais de signe de compute_pvalues (l.204 et l.210-215).
#   --drop-subjects 10 : ttest.py l.31, np.delete(X, 9, 0).
#
# Pendant de la version TEXTE DE LA THESE (§1.2.9 : niveau sujet, 1000 perms),
# deja calculee dans fig3_these/. Le code et le texte ne decrivent pas le meme
# test, on produit les deux.
#
# Pas de --overwrite : le dossier est neuf, et sans le drapeau le script s'arrete
# proprement si le fichier existe deja.
set -euo pipefail

module load python/3.11 2>/dev/null || true
source /home/alouis/mne_env/bin/activate
export PATH=/home/alouis/mne_env/bin:$PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /home/alouis/dream-recall-alex

SAVE=/scratch/alouis/dream_features_noica_1000hz
CORR=/scratch/alouis/dream_features_noica_1000hz_corrected
FIG=/home/alouis/dream-recall-alex/figures
STATE=S2

# N_PERM=1000 sbatch batch_fig3_code.sh   -> premier passage, ~10x plus rapide.
N_PERM=${N_PERM:-9999}
OUT=$CORR/fig3_code_p$N_PERM

mkdir -p "$FIG" /scratch/alouis/logs_dream

echo "=== 1/2 : t-test niveau epoch, schema du code publie, n_perm=$N_PERM ==="
python recompute_ttest_fig3.py \
    --save-path "$SAVE" \
    --out-dir   "$OUT" \
    --state "$STATE" --level epoch --n-perm "$N_PERM" --zscore none \
    --maxstat-scope electrodes --drop-subjects 10 --arthur-pval-bug \
    --n-jobs "$SLURM_CPUS_PER_TASK"

echo "=== 2/2 : figure ==="
python plot_fig3_arthur_topomaps.py \
    --save-path "$SAVE" \
    --in-dir    "$OUT" \
    --results   "$SAVE/results" \
    --coord-file coord_cart_new.txt \
    --state "$STATE" --out "$FIG/fig3_code_${STATE}_p${N_PERM}.png"

echo "=== termine ==="
ls -lh "$FIG/fig3_code_${STATE}_p${N_PERM}.png"
