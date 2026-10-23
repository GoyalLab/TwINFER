#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH -t 4:00:00
#SBATCH --array=0-1
#SBATCH --job-name=fmvar_qc
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm_variants/logs/qc_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm_variants/logs/qc_%A_%a.err
# FM03 (9.9M barcode rows) and FM07 (18.3M rows): singletCode on the entire sheet. FM05 was already built locally.
set -euo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
export TWINFER_PROJECT_ROOT=$ROOT TWINFER_CODE_ROOT=$ROOT/code/TwINFER
# conda hook fails in batch: env python directly (no activation)
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
export PYTHONPATH=$ROOT/code/TwINFER/package:$ROOT/code/TwINFER:${PYTHONPATH:-}
DS=(FM03 FM07)
cd $ROOT/code/TwINFER/paper_analysis/fatemap_pipeline
echo "[$(date)] ${DS[$SLURM_ARRAY_TASK_ID]} QC"
"$PY" build_fm_variants_qc_matrix.py "${DS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] done"
