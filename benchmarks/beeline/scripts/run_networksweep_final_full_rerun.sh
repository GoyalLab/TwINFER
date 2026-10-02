#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem 64GB
#SBATCH -t 1:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs/full_rerun_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs/full_rerun_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Full, clean rerun of 5 of the 7 BEELINE algorithms (PIDC, PPCOR, SCODE,
# SCSGL, PEARSON) against the finalized, SCODE-anchor-verified 150-file
# manifest (inputs/network_sweep_final_20260824,
# config-files/config_networksweep_final_20260824.yaml). GENIE3 and GRNBOOST2
# are EXCLUDED here (should_run: False in that config) -- they run separately,
# sequentially, via run_genie3_grnboost2_sequential.sh, since both are
# Dask/arboreto-based and running them at this job's 15-way dataset
# parallelism risks many concurrent Dask processes hitting peak memory at
# once (see that script's own header for the OOM history). Supersedes the
# earlier patchwork of partial GENIE3/PPCOR/PEARSON fixes on mixed-provenance
# caches with one self-consistent, timestamped artifact.
#
# One BLRunner.py process per dataset (15-way parallel, same pattern as
# run_ppcor_rerun.sh / run_scode_rep6_chunk.sh). SCODE is the long pole
# (~51s/run x 10 runs/dataset =~ 510s per dataset process); everything else
# adds well under that per dataset, so total wall time should land around
# 10-15 minutes plus environment/module overhead. None of these 5 algorithms
# showed any memory-pressure signal in the original run, so 64GB across the
# 15 concurrent (lightweight) processes is ample headroom, not a tight fit.
#
# Usage: sbatch run_networksweep_final_full_rerun.sh

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_BEELINE_PATH}
BASE_CONFIG="${SCRIPT_DIR}/config-files/config_networksweep_final_20260824.yaml"
SUBCONFIG_DIR="${SCRIPT_DIR}/config-files/_networksweep_final_20260824_per_dataset"
# LOG_DIR="/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs"   # [2026-09-30 replaced by env.sh variable]
LOG_DIR="${TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/beeline_inference/logs"
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
        --yes \
        > "${LOG_DIR}/full_${dataset_id}.out" 2> "${LOG_DIR}/full_${dataset_id}.err" &
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
    echo "[$(date)] One or more per-dataset runs failed -- check ${LOG_DIR}/full_*.err"
    exit 1
fi

echo "[$(date)] Full BEELINE rerun finished."
