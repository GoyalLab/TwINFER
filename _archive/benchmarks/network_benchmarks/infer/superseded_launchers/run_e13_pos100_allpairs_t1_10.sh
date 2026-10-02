#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 8GB
#SBATCH -t 1:00:00
#SBATCH --job-name=e13_pos100_allpairs_t1_10
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/e13_pos100_allpairs_t1_10_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/e13_pos100_allpairs_t1_10_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# t1=10,t2=20 rerun of infer_e13_pos100_allpairs.py (measure metrics farther from the division
# event). 30 replicates, tested at ~55s/replicate locally -> ~28min sequential, 1h budget for
# headroom. Output: analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs_t1_10/

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/infer"

echo "[$(date)] e13_pos100 t1=10 ALL_PAIRS rerun (30 replicates)"
# "$PYTHON" infer_e13_pos100_allpairs_t1_10.py --n-cores 4   # [2026-09-30 merged: the _t1_10 script is now the canonical script with TWINFER_T1_VARIANT=t1_10]
TWINFER_T1_VARIANT=t1_10 "$PYTHON" infer_e13_pos100_allpairs.py --n-cores 4
echo "[$(date)] Done"
