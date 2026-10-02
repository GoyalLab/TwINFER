#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=18
#SBATCH --mem 24GB
#SBATCH -t 1:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs/final_retry_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs/final_retry_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Final cleanup pass for the ~18 SCODE nRep=6 runs left over from
# stale/partial working_dirs (from the very first killed attempt) that
# --skip-populated wrongly treated as already-done. No --skip-populated here
# so each Runner.__init__ unconditionally erases its working_dir first.

# CONFIG_LIST=/home/gzu5140/TwINFER_KA/code/Beeline/config-files/_scode_rep6_final_retry/configs.txt   # [2026-09-30 replaced by env.sh variable]
CONFIG_LIST=${TWINFER_BEELINE_PATH}/config-files/_scode_rep6_final_retry/configs.txt
# LOG_DIR=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

module load R/4.2.3 ruby/3.1.0-gcc-4.8.5
# cd /home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_BEELINE_PATH}

run_one() {
    local cfg="$1"
    local name
    name="$(basename "$cfg" .yaml)"
    "$BEELINE_PYTHON" BLRunner.py --config "$cfg" --yes \
        > "${LOG_DIR}/${name}_finalretry.out" 2> "${LOG_DIR}/${name}_finalretry.err"
}
export -f run_one
export BEELINE_PYTHON LOG_DIR

echo "[$(date)] Processing $(wc -l < "$CONFIG_LIST") remaining configs..."
xargs -a "$CONFIG_LIST" -P 18 -I CFG bash -c 'run_one "$@"' _ CFG
echo "[$(date)] Done."
