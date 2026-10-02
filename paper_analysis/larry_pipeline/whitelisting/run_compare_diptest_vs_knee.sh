#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 1:00:00
#SBATCH --job-name=larry_diptest_vs_knee
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/diptestvknee_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/diptestvknee_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}

echo "[$(date)] Comparing knee(v2) vs dip-test+Otsu whitelist, full pipeline"
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] "$PYTHON" code/TwINFER/paper_analysis/larry_pipeline/whitelisting/compare_diptest_vs_knee_pipeline.py
"$PYTHON" ${TWINFER_CODE_ROOT}/paper_analysis/larry_pipeline/whitelisting/compare_diptest_vs_knee_pipeline.py
echo "[$(date)] Done"
