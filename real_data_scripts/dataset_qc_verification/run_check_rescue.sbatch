#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rescue_check
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rescue_check_%j.out
# criterion-4 rescue: as implemented (cellID pooled across samples) vs corrected (per sample+cellID); one process per dataset
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3; L=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs
for ds in FM01 FM05 FM06 FM07 FM08 Watermelon LARRY; do $PY check_rescue_logic.py $ds > $L/rescue_${ds}_${SLURM_JOB_ID}.log 2>&1 & done
wait
grep -h '^{' $L/rescue_*_${SLURM_JOB_ID}.log
