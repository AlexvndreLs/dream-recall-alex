#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=thr_ica
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/thresholds_%j.out
#SBATCH --error=logs/thresholds_%j.err

set -euo pipefail
cd ~/dream-recall-alex
source /home/alouis/mne_env/bin/activate

BIDS=/home/alouis/scratch/dream_bids
DERIV=$BIDS/derivatives

echo "=== Test 1 sujet (sub-05) ==="
python -c "
from analyze_thresholds import scores_for_subject
from pathlib import Path
rows = scores_for_subject('05', Path('$BIDS'), Path('$DERIV'), with_iclabel=False)
assert rows, 'ICA absente ?'
print('n composantes:', len(rows), '| comp 0:', rows[0])
"

echo "=== Balayage complet 38 sujets (--iclabel) ==="
python analyze_thresholds.py \
    --bids-path  "$BIDS" \
    --deriv-root "$DERIV" \
    --out-dir    ./threshold_analysis --iclabel

echo "=== Termine ==="
ls -lh ./threshold_analysis
