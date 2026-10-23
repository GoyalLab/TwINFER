#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 16GB
#SBATCH -t 2:00:00
#SBATCH --array=0-30
#SBATCH --job-name=pairdist
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/pairdist_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/pairdist_%A_%a.err
# clonal vs random pair PCA distance (all 50 PCs). Needs the pca5k outputs -> submit with --dependency=afterok:<pca5k and petrace job ids>.
# HS054_Sot48h and PeTrace_E7.5-R1 were already run locally. Michaels has no clones. Then: python3 pair_distance.py combine
set -euo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
DS=(FM01 FM03 FM05 FM06 FM07 FM08 Watermelon_naive Watermelon_lag Watermelon_late hPSC_20260927 hPSC_endo_yscher hPSC_undifferentiated hPSC_Novo_Nordisk LARRY CellTag3 SpaceBar DTT_mouse PeTrace_E7.5-R2 PeTrace_E7.5-R3 PeTrace_E8.0-R1 PeTrace_E8.0-R2 PeTrace_E8.0-R3 PeTrace_E8.5-R1 PeTrace_E8.5-R2 PeTrace_E8.5-R3 PeTrace_E9.0-R1 PeTrace_E9.0-R2 PeTrace_E9.0-R3 PeTrace_E9.5-R1 PeTrace_E9.5-R2 PeTrace_E9.5-R3)
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
echo "[$(date)] ${DS[$SLURM_ARRAY_TASK_ID]}"
"$PY" pair_distance.py "${DS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] done"
