#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16GB
#SBATCH -t 0:30:00
#SBATCH --job-name=competitors_yscher
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/competitors_yscher_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/competitors_yscher_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}

: "${CRITERION:?CRITERION must be set, e.g. sbatch --export=ALL,CRITERION=correlation run_competitors_yscher_single.sh}"

for LEVEL in high mid low; do
    GENE_SET="${CRITERION}_${LEVEL}"
    echo "[$(date)] Competitor GRN methods (rho, ppcor, PIDC, GENIE3, GRNBoost2) on yscher's $GENE_SET panel, unrestricted (ALL_GENES=1), cp10k-normalized, pooled day2+4"
# /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_competitor_methods.py   # [2026-09-30 replaced by env.sh variable]
    GENE_SET="$GENE_SET" N_CORES=8 ALL_GENES=1 INPUT_DIR=twinfer_input_yscher_cp10k NETWORKS_DIR_NAME=networks_yscher "$PYTHON" \
        ${TWINFER_CODE_ROOT}/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_competitor_methods.py
    echo "[$(date)] Done ($GENE_SET)"
done
