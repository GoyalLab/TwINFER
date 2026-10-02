#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem 150GB
#SBATCH -t 1:00:00
#SBATCH --job-name=real_networks
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference/logs/chunk_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference/logs/chunk_%j.err
set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Runs all 7 enabled BEELINE algorithms (PIDC, GENIE3, GRNBOOST2, PPCOR, SCODE
# nRep=6, SCSGL, PEARSON) against B_cell_activation and Circadian_cycle, one
# BLRunner.py process per (dataset, run) config, fanned out via xargs -P --
# same pattern as run_scode_rep6_chunk.sh, generalized to all methods (hence
# the extra julia module load, needed for PIDC).
#
# Usage: sbatch run_real_networks_chunk.sh <chunk_list_file> [concurrency]
#
# concurrency (optional, default 40): xargs -P value. Lower this for chunks
# containing memory-heavy algorithms (GENIE3/GRNBOOST2 via arboreto) on
# larger networks -- e.g. a 20-config, 20-way run of EMT/Pluripotent (17/36
# genes) OOM-killed the whole job at --mem 150G because GENIE3 alone peaks
# at ~10-26GB per process and enough of the 20 processes hit their GENIE3
# step at the same time to exceed it.

CHUNK_FILE="$1"
CONCURRENCY="${2:-40}"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference/logs"
# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
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

echo "[$(date)] Processing $(wc -l < "$CHUNK_FILE") configs from $CHUNK_FILE at ${CONCURRENCY}-way concurrency..."

xargs -a "$CHUNK_FILE" -P "$CONCURRENCY" -I CFG bash -c 'run_one "$@"' _ CFG

echo "[$(date)] Chunk finished."
