#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem 32GB
#SBATCH -t 0:45:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs/pidc_fix_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs/pidc_fix_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# PIDC-only rerun, all 15 datasets, 15-way parallel -- fixes the missing
# `module load julia/1.10.2` that caused every PIDC run in the original full
# rerun to fail silently. See config_pidc_fix_20260824.yaml.

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
BASE_CONFIG="${SCRIPT_DIR}/config-files/config_pidc_fix_20260824.yaml"
SUBCONFIG_DIR="${SCRIPT_DIR}/config-files/_pidc_fix_20260824_per_dataset"
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$SUBCONFIG_DIR" "$LOG_DIR"
module load julia/1.10.2

cd "$SCRIPT_DIR"

echo "[$(date)] julia on PATH: $(which julia)"
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

echo "[$(date)] Launching one BLRunner.py per dataset (15 parallel processes)..."
declare -a PIDS=()
for dataset_id in $DATASET_IDS; do
    "$BEELINE_PYTHON" BLRunner.py \
        --config "${SUBCONFIG_DIR}/${dataset_id}.yaml" \
        --yes \
        > "${LOG_DIR}/pidc_${dataset_id}.out" 2> "${LOG_DIR}/pidc_${dataset_id}.err" &
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
    echo "[$(date)] One or more per-dataset runs failed -- check ${LOG_DIR}/pidc_*.err"
    exit 1
fi

echo "[$(date)] PIDC fix run finished."
