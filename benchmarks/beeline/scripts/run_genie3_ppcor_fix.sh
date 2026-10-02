#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 64GB
#SBATCH -t 0:30:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference/logs/genie3_ppcor_fix_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference/logs/genie3_ppcor_fix_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Fixes two source-file problems found by cross-checking every algorithm's
# cached working_dir/ExpressionData.csv against the SCODE anchor (SCODE never
# rerun, slowest, treated as ground truth for "which simulation replicate was
# actually used"):
#   1. GENIE3 -- 9 (dataset, run) slots were OOM-killed by arboreto/Dask under
#      the original (much smaller) memory allocation. This job's 64GB should
#      be more than sufficient (the killed runs were only ~4GB resident).
#   2. PPCOR -- 13 (dataset, run) slots whose cached ExpressionData.csv came
#      from a DIFFERENT simulation retry than every other algorithm, because
#      run_ppcor_rerun.sh (a prior, unrelated p-value-prefiltering fix) reran
#      PPCOR hours after the others had already cached their inputs, by which
#      point the canonical converter input had drifted for those slots.
# Both configs target only the specific datasets involved (4 for GENIE3, 5 for
# PPCOR) with scan_run_subdirectories -- every run within a targeted dataset
# gets regenerated, which harmlessly reproduces identical output for the
# labels that were already correct (same staged source file either way).
#
# Usage: sbatch run_genie3_ppcor_fix.sh

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_06082026/beeline_inference/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$LOG_DIR"
module load R/4.2.3
cd "$SCRIPT_DIR"

echo "[$(date)] Running GENIE3 fix (config_genie3_rerun.yaml, 4 datasets)..."
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_genie3_rerun.yaml --yes \
    > "${LOG_DIR}/genie3_fix.out" 2> "${LOG_DIR}/genie3_fix.err"
echo "[$(date)] GENIE3 fix done."

echo "[$(date)] Running PPCOR source fix (config_ppcor_source_fix.yaml, 5 datasets)..."
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_ppcor_source_fix.yaml --yes \
    > "${LOG_DIR}/ppcor_source_fix.out" 2> "${LOG_DIR}/ppcor_source_fix.err"
echo "[$(date)] PPCOR source fix done."

echo "[$(date)] Both fixes complete."
