#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 48GB
#SBATCH -t 2:00:00
#SBATCH --array=0-17
#SBATCH --job-name=pca5k
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/pca5k_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/pca5k_%A_%a.err
# 5k-HVG PCA + elbow plot per dataset (HS054_Sot48h and PeTrace_E7.5-R1 already done locally);
# afterwards run `python3 pca5k.py combine` for the all-dataset figure
set -euo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
DS=(FM01 FM03 FM05 FM06 FM07 FM08 Watermelon_naive Watermelon_lag Watermelon_late hPSC_20260927 hPSC_endo_yscher hPSC_undifferentiated hPSC_Novo_Nordisk LARRY Michaels_et_al CellTag3 SpaceBar DTT_mouse)
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
echo "[$(date)] ${DS[$SLURM_ARRAY_TASK_ID]}"
"$PY" pca5k.py "${DS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] done"
