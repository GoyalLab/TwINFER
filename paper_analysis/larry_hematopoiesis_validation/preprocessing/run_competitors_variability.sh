#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 4:00:00
#SBATCH --job-name=competitors_variability
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/competitors_variability_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/competitors_variability_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}

for GENE_SET in variability_high variability_mid variability_low; do
    echo "[$(date)] Competitor GRN methods (rho, ppcor, PIDC, GENIE3, GRNBoost2) on $GENE_SET, pooled day2+4"
# /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_competitor_methods.py   # [2026-09-30 replaced by env.sh variable]
    GENE_SET="$GENE_SET" N_CORES=8 "$PYTHON" \
        ${TWINFER_CODE_ROOT}/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_competitor_methods.py
    echo "[$(date)] Done ($GENE_SET)"
done
