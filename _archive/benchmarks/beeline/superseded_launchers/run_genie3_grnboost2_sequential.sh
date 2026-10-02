#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 1:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs/genie3_grnboost2_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs/genie3_grnboost2_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# GENIE3 + GRNBOOST2, all 150 runs, ONE BLRunner.py process (no per-dataset
# parallelism) -- see config_genie3_grnboost2_sequential.yaml for why. 32GB is
# generous headroom for a single Dask/arboreto process on 6-gene x 12000-cell
# data (the original OOM kill was ~4.3GB resident under a much smaller limit).
#
# Usage: sbatch run_genie3_grnboost2_sequential.sh

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "[$(date)] Running GENIE3 + GRNBOOST2 sequentially across all 15 datasets (150 runs each)..."
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_genie3_grnboost2_sequential.yaml --yes \
    > "${LOG_DIR}/genie3_grnboost2.out" 2> "${LOG_DIR}/genie3_grnboost2.err"

echo "[$(date)] GENIE3 + GRNBOOST2 sequential run finished."
