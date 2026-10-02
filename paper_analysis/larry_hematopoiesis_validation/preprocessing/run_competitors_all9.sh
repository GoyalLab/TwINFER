#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 4:00:00
#SBATCH --job-name=competitors_all9
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/competitors_all9_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/competitors_all9_%A_%a.err
#SBATCH --array=0-8
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# rho / ppcor / PIDC / GENIE3 / GRNBoost2 on all 9 gene sets, pooled day2+4, INPUT_DIR=twinfer_input_cp10k
# (log1p-CP10k normalized panels). Writes resources/networks/{method}_{GENE_SET}.csv.
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}

GENE_SETS=(correlation_high correlation_mid correlation_low \
           detection_high detection_mid detection_low \
           variability_high variability_mid variability_low)
GENE_SET="${GENE_SETS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] competitors on $GENE_SET (cp10k input)"
# /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_competitor_methods.py   # [2026-09-30 replaced by env.sh variable]
INPUT_DIR=twinfer_input_cp10k GENE_SET="$GENE_SET" N_CORES=8 "$PYTHON" \
    ${TWINFER_CODE_ROOT}/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_competitor_methods.py
echo "[$(date)] Done ($GENE_SET)"
