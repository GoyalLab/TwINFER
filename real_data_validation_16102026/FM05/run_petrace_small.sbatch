#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics-himem
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --array=0-7
#SBATCH --job-name=petrace_small
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/petrace_small_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/petrace_small_%A_%a.err
# PeTrace (final .h5td only, no raw): final-only QC metrics + 5k-HVG PCA per replicate.
# (PeTrace_E7.5-R1 already done locally.)
set -euo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
DS=(PeTrace_E7.5-R2 PeTrace_E7.5-R3 PeTrace_E8.0-R1 PeTrace_E8.0-R2 PeTrace_E8.0-R3 PeTrace_E8.5-R1 PeTrace_E8.5-R2 PeTrace_E8.5-R3)
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
echo "[$(date)] ${DS[$SLURM_ARRAY_TASK_ID]}"
"$PY" qc_verify.py "${DS[$SLURM_ARRAY_TASK_ID]}"
"$PY" pca5k.py "${DS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] done"
