#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --mem 64GB
#SBATCH -t 2:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/twinfer_inference/logs/rerun_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/twinfer_inference/logs/rerun_%j.err
set -euo pipefail

# TwINFER inference over every finished mixed_network_sweep simulation
# (infer_mixed_network_sweep.py -- same infer_with_twinfer settings as the
# network_sweep_final_20260824 rerun, trimmed to the current package
# signature). Measured ~100 s per n=10 inference at 8 cores; ~30-40 s for
# n=6. 10 long-lived parallel batches, each handling its share of the
# discovered simulations, pays the import/JIT cost once per batch.
#
# Idempotent: skips any (net, rep) whose *_all_results.json already exists,
# so rerun this repeatedly as more simulations finish.
#
# Usage: sbatch run_infer_mixed_network_sweep.sh

SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/TwINFER/synthetic_network_analysis
LOG_DIR=/home/gzu5140/TwINFER_KA/analysis_data/mixed_network_sweep/twinfer_inference/logs
PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
N_BATCHES=10

echo "[$(date)] Discovering tasks and building ${N_BATCHES} batches..."
mapfile -t TASKS < <("$PYTHON" - <<'PYEOF'
import infer_mixed_network_sweep as M
import os
for net, k, _ in M.discover_tasks():
    if not os.path.exists(os.path.join(M.OUTPUT_DIR, f"{net}_rep{k}_all_results.json")):
        print(f"{net}:{k}")
PYEOF
)
echo "[$(date)] ${#TASKS[@]} simulation(s) need inference."
if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

declare -a PIDS=()
for ((b = 0; b < N_BATCHES; b++)); do
    BATCH=()
    for ((i = b; i < ${#TASKS[@]}; i += N_BATCHES)); do
        BATCH+=("${TASKS[$i]}")
    done
    [ "${#BATCH[@]}" -eq 0 ] && continue
    ONLY=$(IFS=,; echo "${BATCH[*]}")
    "$PYTHON" infer_mixed_network_sweep.py --n-cores 4 --only "$ONLY" \
        > "${LOG_DIR}/batch_${b}.out" 2> "${LOG_DIR}/batch_${b}.err" &
    PIDS+=($!)
    echo "  started batch ${b} (pid $!, ${#BATCH[@]} tasks)"
done

echo "[$(date)] Waiting for ${#PIDS[@]} batch(es)..."
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=1
done
[ "$FAILED" -ne 0 ] && { echo "[$(date)] a batch failed -- see ${LOG_DIR}/batch_*.err"; exit 1; }
echo "[$(date)] TwINFER inference over mixed_network_sweep finished."
