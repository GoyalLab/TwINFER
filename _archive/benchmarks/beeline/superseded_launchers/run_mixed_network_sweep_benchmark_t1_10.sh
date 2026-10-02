#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem 96GB
#SBATCH -t 4:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/beeline_inference_t1_10/logs/benchmark_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/beeline_inference_t1_10/logs/benchmark_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# t1=10,t2=20 rerun of run_mixed_network_sweep_benchmark.sh: PIDC/PPCOR/SCODE/SCSGL/PEARSON over
# the twin_paired-only mixed_network_sweep_t1_10 input root (144 replicates / 48 datasets).
# GENIE3/GRNBOOST2 excluded here -- run separately via
# run_mixed_network_sweep_genie3_grnboost2_t1_10.sh (Dask/arboreto memory safety). Same
# one-BLRunner-process-per-dataset xargs fanout as the original. Wall time trimmed from 6h to 4h
# since this root has ONLY twin_paired (no spread duplicate).
#
# Regenerate the config first if the dataset set changed:
#   python generate_t1_10_beeline_configs.py
#
# Usage: sbatch run_mixed_network_sweep_benchmark_t1_10.sh [concurrency]

CONCURRENCY="${1:-15}"
# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
BASE_CONFIG="${SCRIPT_DIR}/config-files/config_mixed_network_sweep_t1_10.yaml"
SUBCONFIG_DIR="${SCRIPT_DIR}/config-files/_mixed_network_sweep_t1_10_per_dataset"
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/beeline_inference_t1_10/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/mixed_network_sweep/beeline_inference_t1_10/logs"
# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

mkdir -p "$SUBCONFIG_DIR" "$LOG_DIR"
module load julia/1.10.2 R/4.2.3 ruby/3.1.0-gcc-4.8.5
cd "$SCRIPT_DIR"

echo "[$(date)] Generating one per-dataset config under ${SUBCONFIG_DIR} ..."
DATASET_IDS=$("$BEELINE_PYTHON" - "$BASE_CONFIG" "$SUBCONFIG_DIR" <<'PYEOF'
import sys, yaml
base_config_path, subconfig_dir = sys.argv[1], sys.argv[2]
with open(base_config_path) as f:
    config = yaml.safe_load(f)
dataset_ids = [d["dataset_id"] for d in config["input_settings"]["datasets"]]
for target_id in dataset_ids:
    variant = yaml.safe_load(yaml.dump(config))
    for d in variant["input_settings"]["datasets"]:
        d["should_run"] = [d["dataset_id"] == target_id]
    with open(f"{subconfig_dir}/{target_id}.yaml", "w") as f:
        yaml.dump(variant, f, sort_keys=False)
    print(target_id)
PYEOF
)

echo "[$(date)] Launching BLRunner.py per dataset at ${CONCURRENCY}-way concurrency..."
run_one() {
    local dataset_id="$1"
    "$BEELINE_PYTHON" BLRunner.py \
        --config "config-files/_mixed_network_sweep_t1_10_per_dataset/${dataset_id}.yaml" \
        --yes --skip-populated \
        > "${LOG_DIR}/full_${dataset_id}.out" 2> "${LOG_DIR}/full_${dataset_id}.err"
}
export -f run_one
export BEELINE_PYTHON LOG_DIR

printf '%s\n' $DATASET_IDS | xargs -P "$CONCURRENCY" -I DS bash -c 'run_one "$@"' _ DS

echo "[$(date)] mixed_network_sweep t1=10 BEELINE benchmark finished."
