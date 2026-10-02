#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference/logs/genie3_grnboost2_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference/logs/genie3_grnboost2_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# GENIE3 + GRNBOOST2 for the 3 e13_pos100 datasets, single BLRunner.py process
# (Dask/arboreto memory safety, same reasoning as
# run_mixed_network_sweep_genie3_grnboost2.sh) -- tiny job (3 datasets x 10
# reps x 2 schemes = 60 runs).
#
# Usage: sbatch run_e13_pos100_genie3_grnboost2.sh

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_inference/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$LOG_DIR"
module load R/4.2.3 ruby/3.1.0-gcc-4.8.5
cd "$SCRIPT_DIR"

echo "[$(date)] Running GENIE3 + GRNBOOST2 for e13_pos100..."
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_e13_pos100_genie3_grnboost2.yaml --yes --skip-populated \
    > "${LOG_DIR}/genie3_grnboost2.out" 2> "${LOG_DIR}/genie3_grnboost2.err"

echo "[$(date)] e13_pos100 GENIE3 + GRNBOOST2 finished."
