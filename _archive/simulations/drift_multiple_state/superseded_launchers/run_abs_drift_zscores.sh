#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --mem 64GB
#SBATCH -t 4:00:00
#SBATCH --job-name=abs_drift_zscores
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/drift_inference/logs/abs_drift_zscores_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/drift_inference/logs/abs_drift_zscores_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/simulations/drift_multiple_state"

# mkdir -p /home/gzu5140/TwINFER_KA/analysis_data/drift_inference/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/logs
# OUT_DIR=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/eight_scenario/t1_1_t2_20/abs_drift   # [2026-09-30 replaced by env.sh variable]
OUT_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario/t1_1_t2_20/abs_drift

"$PYTHON" compute_abs_drift_zscores.py --output-dir "$OUT_DIR" --n-draws 10000 --jobs 40
