#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem 64GB
#SBATCH -t 2:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs/chunk_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs/chunk_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Processes one chunk-list file of per-run SCODE nRep=6 configs (one config per
# remaining (dataset, run) pair -- generated ad hoc, not scan_run_subdirectories)
# with 45-way concurrency via xargs -P. Submitted twice (two chunk files, two
# jobs on two separate nodes) to get ~90-way parallelism total across the
# ~251 runs still remaining after the first (15-way, per-dataset) attempt --
# that one measured ~408s/run in practice, so 90-way parallelism targets
# ~251/90 * 408s =~ 19 min per job, comfortably under the requested ~30 min.
#
# Usage: sbatch run_scode_rep6_chunk.sh <chunk_list_file>

CHUNK_FILE="$1"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs"
# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}

mkdir -p "$LOG_DIR"
module load R/4.2.3 ruby/3.1.0-gcc-4.8.5
cd "$SCRIPT_DIR"

run_one() {
    local cfg="$1"
    local name
    name="$(basename "$cfg" .yaml)"
    "$BEELINE_PYTHON" BLRunner.py --config "$cfg" --yes --skip-populated \
        > "${LOG_DIR}/${name}.out" 2> "${LOG_DIR}/${name}.err"
}
export -f run_one
export BEELINE_PYTHON LOG_DIR

echo "[$(date)] Processing $(wc -l < "$CHUNK_FILE") configs from $CHUNK_FILE at 45-way concurrency..."

xargs -a "$CHUNK_FILE" -P 45 -I CFG bash -c 'run_one "$@"' _ CFG

echo "[$(date)] Chunk finished."
