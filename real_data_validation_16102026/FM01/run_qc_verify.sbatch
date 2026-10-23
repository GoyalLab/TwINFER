#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 48GB
#SBATCH -t 2:00:00
#SBATCH --array=0-10
#SBATCH --job-name=qcverify
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/qcverify_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/qcverify_%A_%a.err
# raw-vs-final QC verification + plots. Already done locally (small): HS054_Sot48h, Michaels_et_al, DTT_mouse, PeTrace_E7.5-R1.
# LARRY = pilot (LSK_d2_1 only, from LARRY_data/raw/LSK_d2_1.zip counts). CellTag3 / SpaceBar: final-only (no raw matrix here).
set -euo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # env python directly (conda hook fails in batch)
DS=(FM01 FM03 FM05 FM06 FM07 FM08 Watermelon hPSC_20260927 LARRY CellTag3 SpaceBar)
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
echo "[$(date)] ${DS[$SLURM_ARRAY_TASK_ID]}"
"$PY" qc_verify.py "${DS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] done"
