#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16GB
#SBATCH -t 0:20:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/beeline_inference/logs/pidc_fix_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/beeline_inference/logs/pidc_fix_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/autoregulation_benchmark_20260824/beeline_inference/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$LOG_DIR"
module load julia/1.10.2
cd "$SCRIPT_DIR"

echo "[$(date)] julia on PATH: $(which julia)"
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_autoreg_pidc_fix.yaml --yes \
    > "${LOG_DIR}/pidc_fix.out" 2> "${LOG_DIR}/pidc_fix.err"

echo "[$(date)] PIDC autoregulation fix finished."
