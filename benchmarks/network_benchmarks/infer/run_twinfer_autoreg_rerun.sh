#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/twinfer_inference/logs/rerun_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/twinfer_inference/logs/rerun_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Full TwINFER rerun for the 40-file autoregulation manifest (4 topologies x
# 10 replicates), same batching approach as run_twinfer_150_rerun.sh.

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/TwINFER/synthetic_network_analysis   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR="${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/infer"
# LOG_DIR=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/twinfer_inference/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/autoregulation_benchmark_20260824/twinfer_inference/logs
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# MANIFEST=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/source_manifest_autoreg.json   # [2026-09-30 replaced by env.sh variable]
MANIFEST=${TWINFER_PROJECT_ROOT}/analysis_data/autoregulation_benchmark_20260824/source_manifest_autoreg.json
# OUTPUT_DIR=/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/twinfer_inference   # [2026-09-30 replaced by env.sh variable]
OUTPUT_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/autoregulation_benchmark_20260824/twinfer_inference

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "[$(date)] Building task batches..."
BATCHES=$("$PYTHON" - "$MANIFEST" <<'PYEOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
manifest.pop("logs", None)
tasks = [f"{ds}:{label}" for ds, labels in manifest.items() for label in labels if "source" in labels[label]]
N_BATCHES = 8
batches = [tasks[i::N_BATCHES] for i in range(N_BATCHES)]
for b in batches:
    if b:
        print(",".join(b))
PYEOF
)

echo "[$(date)] Launching parallel batches..."
declare -a PIDS=()
i=0
while IFS= read -r batch; do
    i=$((i+1))
    "$PYTHON" rerun_twinfer_150.py --n-cores 4 --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" --only "$batch" \
        > "${LOG_DIR}/batch_${i}.out" 2> "${LOG_DIR}/batch_${i}.err" &
    PIDS+=($!)
    echo "  started batch ${i} (pid $!, $(echo "$batch" | tr ',' '\n' | wc -l) tasks)"
done <<< "$BATCHES"

echo "[$(date)] Waiting for all ${#PIDS[@]} batches to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "[$(date)] One or more batches failed -- check ${LOG_DIR}/batch_*.err"
    exit 1
fi

echo "[$(date)] TwINFER autoregulation rerun finished."
