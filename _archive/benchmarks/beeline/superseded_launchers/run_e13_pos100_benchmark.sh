#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference/logs/benchmark_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference/logs/benchmark_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# BEELINE benchmark for the e13_pos100 datasets (3 of the network_sweep_final
# OFAT "center" topologies, simulated separately and never run through
# inference): PIDC, PPCOR, SCODE, SCSGL, PEARSON (one BLRunner.py process per
# dataset, 3-way concurrency -- tiny job, same pattern as
# run_mixed_network_sweep_benchmark.sh). GENIE3/GRNBOOST2 run separately via
# run_e13_pos100_genie3_grnboost2.sh.
#
# Usage: sbatch run_e13_pos100_benchmark.sh

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
CONFIG="${SCRIPT_DIR}/config-files/config_e13_pos100.yaml"
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_inference/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$LOG_DIR"
module load julia/1.10.2 R/4.2.3 ruby/3.1.0-gcc-4.8.5
cd "$SCRIPT_DIR"

echo "[$(date)] Running e13_pos100 BEELINE benchmark (PIDC/PPCOR/SCODE/SCSGL/PEARSON)..."
"$BEELINE_PYTHON" BLRunner.py --config "$CONFIG" --yes --skip-populated \
    > "${LOG_DIR}/benchmark.out" 2> "${LOG_DIR}/benchmark.err"

echo "[$(date)] e13_pos100 BEELINE benchmark finished."
