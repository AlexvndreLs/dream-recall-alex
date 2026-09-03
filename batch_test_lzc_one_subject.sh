#!/bin/bash
#SBATCH --job-name=test_lzc_1sub
#SBATCH --account=rrg-kjerbi
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/test_lzc_1sub_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/test_lzc_1sub_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Remplace l'etape 2 du runbook (le salloc), qui echouait sur Fir avec
# "invalid partition specified: cpularge_interac". Le plugin de routage
# d'Alliance classe les sessions interactives par ratio memoire/coeur, et
# 32 G pour 1 coeur tombait dans une classe de partition inexistante ici.
# Un job batch normal ne passe pas par ce routage.
#
# 32 G reste le bon ordre de grandeur : une nuit a 1000 Hz fait environ 4.4 Go
# pour le raw precharge, plus 3 a 4.4 Go pour le dictionnaire d'epoques, plus
# un pic transitoire au np.stack. C'est la repartition sur un seul coeur qui
# posait probleme, pas le total. Ici 8 coeurs, soit 4 G/coeur, ratio standard.
#
# --n-jobs 1 malgre les 8 coeurs : on veut mesurer le temps par epoque sur un
# seul coeur, c'est ce chiffre qui sert a calibrer --time du job complet.
# Les 8 coeurs ne sont demandes que pour obtenir les 32 G au bon ratio.
#
# Ce que le log doit contenir a la fin :
#   - la version de numba, sinon antropy retombe sur du Python pur et le
#     comptage LZ76 devient inutilisable
#   - le nombre d'epoques et le temps par stade, a multiplier par 38 sujets
#     puis diviser par 8 pour estimer le job complet
#   - la liste des .npz ecrits, qui doivent tous etre en s01 et non s1

SUB=${SUB:-01}
DERIV=/scratch/alouis/dream_bids/derivatives_1000hz/preprocessed-noica
SAVE=/scratch/alouis/dream_features_noica_1000hz

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Start: $(date)"

cd /home/alouis/dream-recall-alex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
source /home/alouis/mne_env/bin/activate

echo "=== 0. dependances ==="
python -c "import numba, antropy, mne; print('numba', numba.__version__, \
'| antropy', antropy.__version__, '| mne', mne.__version__)"

echo "=== 1. le derivative existe-t-il au chemin attendu ? ==="
ls -la "${DERIV}/sub-${SUB}/eeg/" | head

echo "=== 2. extraction LZC, un sujet, un coeur ==="
python feat_extract_lzc.py \
    --deriv-path "${DERIV}" \
    --save-path  "${SAVE}" \
    --subjects   "${SUB}" \
    --n-jobs     1

echo "=== 3. fichiers ecrits ==="
ls -la "${SAVE}/lzc/" | grep "s${SUB}" || echo "AUCUN FICHIER, voir l'erreur ci-dessus"

echo "=== 4. relecture par le meme chemin que classify.py ==="
python -c "
from pathlib import Path
from utils import load_atomic
for st in ['S1','S2','S3','S4','REM']:
    a = load_atomic(Path('${SAVE}'), 'lzc', '${SUB}', st)
    print(f'  {st:4s}', 'absent' if a is None else f'{a.shape}  moyenne={a.mean():.4f}')
"

echo "End: $(date)"
echo
echo "Calibration de batch_feat_extract_lzc.sh : reprendre le temps total"
echo "de l'etape 2, multiplier par 38 sujets, diviser par 8 coeurs, ajouter"
echo "une marge. Ajuster --time en consequence avant de lancer le job complet."