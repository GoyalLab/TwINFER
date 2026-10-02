#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16GB
#SBATCH -t 1:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/code/Beeline/_verify_scoring_work/verify_scoring_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/code/Beeline/_verify_scoring_work/verify_scoring_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# cd /home/gzu5140/TwINFER_KA/code/Beeline/_verify_scoring_work   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_CODE_ROOT}/benchmarks/beeline/verify_scoring
# /home/gzu5140/.conda/envs/twinfer-code/bin/python verify_scoring.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}" verify_scoring.py
