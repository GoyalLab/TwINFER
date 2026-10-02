#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 64GB
#SBATCH -t 6:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/beeline_inference/logs/genie3_grnboost2_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/beeline_inference/logs/genie3_grnboost2_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# GENIE3 + GRNBOOST2 for every mixed_network_sweep dataset, ONE BLRunner.py
# process (no per-dataset parallelism) -- both are Dask/arboreto-based and
# running many concurrently risks peak-memory blowups (see
# run_genie3_grnboost2_sequential.sh's history). n=6/n=10 x 12000 cells is
# light; 64GB is generous headroom for one process.
#
# Regenerate the config first if the dataset set changed:
#   python generate_mixed_network_sweep_configs.py
#
# Usage: sbatch run_mixed_network_sweep_genie3_grnboost2.sh

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/mixed_network_sweep/beeline_inference/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$LOG_DIR"
module load R/4.2.3 ruby/3.1.0-gcc-4.8.5
cd "$SCRIPT_DIR"

echo "[$(date)] Running GENIE3 + GRNBOOST2 sequentially over mixed_network_sweep..."
"$BEELINE_PYTHON" BLRunner.py \
    --config config-files/config_mixed_network_sweep_genie3_grnboost2.yaml \
    --yes --skip-populated \
    > "${LOG_DIR}/genie3_grnboost2.out" 2> "${LOG_DIR}/genie3_grnboost2.err"

echo "[$(date)] GENIE3 + GRNBOOST2 run finished."
