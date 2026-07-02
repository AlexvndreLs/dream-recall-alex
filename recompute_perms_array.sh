#!/bin/bash
#SBATCH --job-name=recompute_perms
#SBATCH --account=def-kjerbi
#SBATCH --array=1-24
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/alouis/logs_dream/recompute_perms-%a_%j.out
#SBATCH --error=/scratch/alouis/logs_dream/recompute_perms-%a_%j.err
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL

KEYS=(cov cosp_delta cosp_theta cosp_alpha cosp_sigma cosp_beta)
STATES=(S2 SWS NREM REM)

KEY_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
STATE_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))
KEY=${KEYS[$KEY_IDX]}
STATE=${STATES[$STATE_IDX]}

echo "=== recompute_perms key=${KEY} state=${STATE} ==="
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

python recompute_perms_synchronized.py \
    --save-path /home/alouis/scratch/dream_features_noica \
    --n-jobs $SLURM_CPUS_PER_TASK \
    --n-perm 1000 \
    --key ${KEY} \
    --state ${STATE}

echo "End: $(date)"
