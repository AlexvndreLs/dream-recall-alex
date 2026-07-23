#!/bin/bash
#SBATCH --job-name=diag_sub
#SBATCH --account=rrg-kjerbi
#SBATCH --exclude=fc30555
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/diag_sub_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

source /home/alouis/mne_env/bin/activate
cd /home/alouis/dream-recall-alex

python3 diag_sub_vs_ratio.py
