#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference_t1_10/logs/benchmark_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference_t1_10/logs/benchmark_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# t1=10,t2=20 rerun of run_e13_pos100_benchmark.sh: PIDC/PPCOR/SCODE/SCSGL/PEARSON over the
# twin_paired-only e13_pos100_t1_10 input root (30 replicates, 3 datasets). GENIE3/GRNBOOST2 run
# separately via run_e13_pos100_genie3_grnboost2_t1_10.sh. Regenerate the config first if the
# dataset set changed: python generate_t1_10_beeline_configs.py
#
# Usage: sbatch run_e13_pos100_benchmark_t1_10.sh

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
CONFIG="${SCRIPT_DIR}/config-files/config_e13_pos100_t1_10.yaml"
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/network_sweep_final/e13_pos100/beeline_inference_t1_10/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_inference_t1_10/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$LOG_DIR"
module load julia/1.10.2 R/4.2.3 ruby/3.1.0-gcc-4.8.5
cd "$SCRIPT_DIR"

echo "[$(date)] Running e13_pos100 t1=10 BEELINE benchmark (PIDC/PPCOR/SCODE/SCSGL/PEARSON)..."
"$BEELINE_PYTHON" BLRunner.py --config "$CONFIG" --yes --skip-populated \
    > "${LOG_DIR}/benchmark.out" 2> "${LOG_DIR}/benchmark.err"

echo "[$(date)] e13_pos100 t1=10 BEELINE benchmark finished."
