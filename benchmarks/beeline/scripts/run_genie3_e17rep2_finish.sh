#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 0:20:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs/genie3_e17rep2_finish_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs/genie3_e17rep2_finish_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Finishes the last 5 GENIE3 runs (grn_n6_e17_pos100_density_rep2, labels 5-9)
# via a proper dedicated-memory SLURM job -- NOT direct Bash execution, which
# already failed once here with OOM (exit 137) from lacking any real memory
# allocation. See config_genie3_grnboost2_e17rep2_finish.yaml.

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "[$(date)] Running GENIE3 for grn_n6_e17_pos100_density_rep2 (all labels; 0-4 already done and will just be harmlessly redone)..."
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_genie3_grnboost2_e17rep2_finish.yaml --yes \
    > "${LOG_DIR}/genie3_e17rep2_finish.out" 2> "${LOG_DIR}/genie3_e17rep2_finish.err"

echo "[$(date)] Finished."
