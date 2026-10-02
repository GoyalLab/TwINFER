#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem 150GB
#SBATCH -t 1:00:00
#SBATCH --job-name=real_network_tp_backfill
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/benchmarking_analysis/twinfer_real/outputs/logs/backfill_chunk_%j.out   # [2026-10-01 was /scratch/gzu5140/twinfer_real/...; #SBATCH cannot use variables]
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/benchmarking_analysis/twinfer_real/outputs/logs/backfill_chunk_%j.err
set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Backfills the missing/incomplete twin_paired BEELINE runs for the curated
# real-world networks (GSD: 0/9 run-dirs done, HSC: 0/10, mCAD: 5/10 partial;
# VSC already complete). One BLRunner.py process per run-dir, fanned out via
# xargs -P -- same pattern as run_real_networks_chunk.sh.
#
# Usage: sbatch run_real_network_twinpaired_backfill_chunk.sh

MANIFEST="config-files/_real_network_tp_backfill_manifest.txt"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"
# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] LOG_DIR="/scratch/gzu5140/twinfer_real/outputs/logs"
LOG_DIR="${TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_real/outputs/logs"
# SCRIPT_DIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}

mkdir -p "$LOG_DIR"
module load julia/1.10.2 R/4.2.3 ruby/3.1.0-gcc-4.8.5
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

echo "[$(date)] Processing $(wc -l < "$MANIFEST") configs from $MANIFEST at 24-way concurrency..."

xargs -a "$MANIFEST" -P 24 -I CFG bash -c 'run_one "$@"' _ CFG

echo "[$(date)] Chunk finished."
