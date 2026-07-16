#!/bin/bash
#SBATCH --account=rrg-kjerbi
#SBATCH --job-name=cnn_s2_test
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=cnn_s2_test_%j.out

source /home/alouis/mne_env/bin/activate
cd ~/scratch/cnn-dream-recall

# TEST : 3 epochs seulement, pour valider GPU + data avant le vrai run.
python run.py --h5 /scratch/alouis/cnn_data/all_S2_noica.h5 --device cuda \
    --epochs 3 --batch-size 128 --max-folds 1 --out res_S2_test.json
