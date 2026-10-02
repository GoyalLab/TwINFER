#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 16GB
#SBATCH -t 0:45:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/cleanup_cyclic_autoreg_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/cleanup_cyclic_autoreg_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Compress before_division checkpoint files for cyclic (g3/g4/g5/cyclic_6_nodes)
# and autoregulation -- ~47.6GB total, all matched (no retries in either
# benchmark), so nothing is deleted, only compressed with verification.

# /home/gzu5140/.conda/envs/twinfer-code/bin/python3 /home/gzu5140/TwINFER_KA/code/Beeline/cleanup_cyclic_autoreg.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}" ${TWINFER_CODE_ROOT}/benchmarks/beeline/scripts/cleanup_cyclic_autoreg.py
