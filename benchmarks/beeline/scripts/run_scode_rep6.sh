#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 16GB
#SBATCH -t 4:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs/scode_rep6_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6/logs/scode_rep6_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Reruns ONLY SCODE, at nRep=6 (BEELINE's own reference default -- see
# config-files/config.yaml -- instead of this sweep's cost-reduced nRep=2),
# against the same 15 network_sweep_final datasets config_networksweep_final.yaml
# scores. Writes to a NEW, separate output root
# (beeline_inference_scode_rep6/, sibling to the existing nRep=2
# beeline_inference/) so the old nRep=2 SCODE results -- and every other
# already-completed algorithm under beeline_inference/ -- stay untouched and
# usable while this rerun is in progress.
#
# BLRunner.py itself processes runners strictly sequentially (no internal
# parallelism), and at nRep=6 a single SCODE run takes ~3 min -- sequential
# across all ~296 dataset/run combinations would take ~15 hours. Instead this
# script fans out one BLRunner.py process PER DATASET (15 datasets -> 15
# parallel processes, each sequential only across its own ~18-20 runs), which
# gets the whole rerun down to roughly the time of the single slowest dataset
# (~45-60 min), not the sum of all of them.
#
# Usage:
#   sbatch run_scode_rep6.sh          # submit as a SLURM job
#   bash run_scode_rep6.sh            # or just run directly (no sbatch needed
#                                      # for this scale -- ~15 lightweight
#                                      # single-core R/Ruby processes)

# NOTE: deliberately hardcoded, not self-located via ${BASH_SOURCE[0]} -- sbatch
# copies the submitted script into a SLURM spool directory
# (/var/spool/slurmd/jobXXXXXXX/) and runs it from there, so BASH_SOURCE-based
# self-location resolves to the spool copy's location, not this script's real
# directory (same footgun documented in run_replicates.sh; breaks under
# `sbatch` even though it works fine under plain `bash run_scode_rep6.sh`).
# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
BASE_CONFIG="${SCRIPT_DIR}/config-files/config_scode_rep6.yaml"
SUBCONFIG_DIR="${SCRIPT_DIR}/config-files/_scode_rep6_per_dataset"
# OUTPUT_ROOT="/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6"   # [2026-09-30 replaced by env.sh variable]
OUTPUT_ROOT="${TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_06082026/beeline_inference_scode_rep6"
LOG_DIR="${OUTPUT_ROOT}/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$SUBCONFIG_DIR" "$LOG_DIR"

module load R/4.2.3 ruby/3.1.0-gcc-4.8.5

cd "$SCRIPT_DIR"

echo "[$(date)] Generating one per-dataset config under ${SUBCONFIG_DIR} ..."
DATASET_IDS=$("$BEELINE_PYTHON" - "$BASE_CONFIG" "$SUBCONFIG_DIR" <<'PYEOF'
import sys
import yaml

base_config_path, subconfig_dir = sys.argv[1], sys.argv[2]
with open(base_config_path) as f:
    config = yaml.safe_load(f)

dataset_ids = [d["dataset_id"] for d in config["input_settings"]["datasets"]]
for target_id in dataset_ids:
    variant = yaml.safe_load(yaml.dump(config))  # deep copy
    for d in variant["input_settings"]["datasets"]:
        d["should_run"] = [d["dataset_id"] == target_id]
    out_path = f"{subconfig_dir}/{target_id}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(variant, f, sort_keys=False)
    print(target_id)
PYEOF
)

echo "[$(date)] Launching one BLRunner.py per dataset (15 parallel processes)..."
declare -a PIDS=()
for dataset_id in $DATASET_IDS; do
    "$BEELINE_PYTHON" BLRunner.py \
        --config "${SUBCONFIG_DIR}/${dataset_id}.yaml" \
        --yes --skip-populated \
        > "${LOG_DIR}/${dataset_id}.out" 2> "${LOG_DIR}/${dataset_id}.err" &
    PIDS+=($!)
    echo "  started ${dataset_id} (pid $!)"
done

echo "[$(date)] Waiting for all ${#PIDS[@]} dataset runs to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "[$(date)] One or more per-dataset SCODE runs failed -- check ${LOG_DIR}/*.err"
    exit 1
fi

echo "[$(date)] All SCODE nRep=6 reruns finished. Output under ${OUTPUT_ROOT}"
