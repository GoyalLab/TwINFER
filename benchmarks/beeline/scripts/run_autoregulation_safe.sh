#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 32GB
#SBATCH -t 0:45:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/beeline_inference/logs/safe_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/beeline_inference/logs/safe_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# PIDC, PPCOR, SCODE, SCSGL, PEARSON for the 4 autoregulation datasets
# (4-way per-dataset parallelism, same pattern as run_ppcor_rerun.sh).

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
BASE_CONFIG="${SCRIPT_DIR}/config-files/config_autoregulation_safe_20260824.yaml"
SUBCONFIG_DIR="${SCRIPT_DIR}/config-files/_autoregulation_safe_per_dataset"
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/autoregulation_benchmark_20260824/beeline_inference/logs"
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
    variant = yaml.safe_load(yaml.dump(config))
    for d in variant["input_settings"]["datasets"]:
        d["should_run"] = [d["dataset_id"] == target_id]
    out_path = f"{subconfig_dir}/{target_id}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(variant, f, sort_keys=False)
    print(target_id)
PYEOF
)

echo "[$(date)] Launching one BLRunner.py per dataset (4 parallel processes)..."
declare -a PIDS=()
for dataset_id in $DATASET_IDS; do
    "$BEELINE_PYTHON" BLRunner.py \
        --config "${SUBCONFIG_DIR}/${dataset_id}.yaml" \
        --yes \
        > "${LOG_DIR}/safe_${dataset_id}.out" 2> "${LOG_DIR}/safe_${dataset_id}.err" &
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
    echo "[$(date)] One or more per-dataset runs failed -- check ${LOG_DIR}/safe_*.err"
    exit 1
fi

echo "[$(date)] Autoregulation safe-algorithm run finished."
