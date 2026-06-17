#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=bids_convert
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-38
#SBATCH --output=logs/convert_%a.txt
#SBATCH --error=logs/convert_%a.txt

mkdir -p logs

source ~/mne_env/bin/activate

python convert_to_bidsv3.py $SLURM_ARRAY_TASK_ID
