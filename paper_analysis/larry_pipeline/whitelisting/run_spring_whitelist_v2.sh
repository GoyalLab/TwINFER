#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=larry_spring_whitelist_v2
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/spring_whitelist_v2_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/spring_whitelist_v2_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}

echo "[$(date)] Building SPRING-style whitelist v2 (tuned to target cell count)"
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] "$PYTHON" code/TwINFER/paper_analysis/larry_pipeline/whitelisting/build_spring_whitelist_v2.py
"$PYTHON" ${TWINFER_CODE_ROOT}/paper_analysis/larry_pipeline/whitelisting/build_spring_whitelist_v2.py
echo "[$(date)] Done"
