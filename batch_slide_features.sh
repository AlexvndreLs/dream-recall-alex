#!/bin/bash
#SBATCH --job-name=slide_figs
#SBATCH --account=rrg-kjerbi
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --exclude=fc30555
#SBATCH --output=logs/slide_figs_%j.out
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
set -euo pipefail

source /home/alouis/mne_env/bin/activate
cd ~/dream-recall-alex
mkdir -p logs

SUB=01
CHAN=Cz
BIDS=/home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica/sub-${SUB}/eeg
VHDR=${BIDS}/sub-${SUB}_task-sleep_proc-clean_eeg.vhdr
# le nom exact du events.tsv varie selon la derivation, on le resout au lieu de le supposer
EVENTS=$(ls ${BIDS}/*events.tsv | head -1)

echo "[info] vhdr   : ${VHDR}"
echo "[info] events : ${EVENTS}"

python3 plot_slide_features.py \
  --fif "${VHDR}" \
  --events "${EVENTS}" \
  --channel ${CHAN} \
  --dur 3.0 \
  --pipeline-dur 30 \
  --sws-tmin 24660.0 --rem-tmin 7800.0 \
  --se-sws-tmin 9780.0 --se-rem-tmin 19200.0 \
  --outdir ./plot_slide_en

echo "[fini] figures dans ./plot_slide_en/"