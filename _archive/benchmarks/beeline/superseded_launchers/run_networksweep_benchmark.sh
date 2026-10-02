#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 32GB
#SBATCH -t 24:00:00
#SBATCH --output=/home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/network_sweep/beeline_inference/logs/benchmark_%j.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/network_sweep/beeline_inference/logs/benchmark_%j.err
set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# mkdir -p /home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/network_sweep/beeline_inference/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/network_sweep/beeline_inference/logs

module load julia/1.10.2 R/4.2.3 ruby/3.1.0-gcc-4.8.5

# cd /home/gzu5140/Keerthana_b1042/TwINFER/code/Beeline   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_BEELINE_PATH}

# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

echo "[$(date)] Starting network-sweep benchmark..."
echo "Using python: $BEELINE_PYTHON"
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_networksweep_final.yaml --yes --skip-populated
status=$?
echo "[$(date)] Benchmark finished with exit code $status"
