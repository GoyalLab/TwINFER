#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --mem 48GB
#SBATCH -t 4:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_full_run_v2/logs/slurm_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_full_run_v2/logs/slurm_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Rerun of run_heterogeneity_v2_slurm_full.sh against a FRESH output
# directory (slurm_full_run_v2, not slurm_full_run) -- process_replicate()
# now also extracts Step 2 diagnostics (null mean/std/median, twin z-score)
# into the saved record, which the old slurm_full_run/*.json files predate.
# run_heterogeneity_v2.py skips any task whose output JSON already exists,
# so reusing slurm_full_run would silently keep the old, incomplete records
# instead of recomputing them -- hence the new directory.
#
# Same batch-of-background-processes pattern as the original full run
# (~1.8-2h estimated at --cpus-per-task=40; -t 4:00:00 leaves headroom).
# Safe to resubmit if it times out.

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/heterogeneity_vs_regulation   # [2026-09-30 replaced by env.sh variable]
# SCRIPT_DIR=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/heterogeneity_vs_regulation   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR="${TWINFER_CODE_ROOT}/paper_analysis/heterogeneity_vs_regulation"
# OUTPUT_DIR=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_full_run_v2/results   # [2026-09-30 replaced by env.sh variable]
OUTPUT_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_full_run_v2/results
# LOG_DIR=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_full_run_v2/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_full_run_v2/logs
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"

N_BATCHES=${SLURM_CPUS_PER_TASK:-40}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "[$(date)] Listing all tasks..."
mapfile -t ALL_TASKS < <("$PYTHON" run_heterogeneity_v2.py --list-tasks)
echo "[$(date)] Got ${#ALL_TASKS[@]} task(s); splitting into ${N_BATCHES} batch(es)."

echo "[$(date)] Launching parallel batches..."
declare -a PIDS=()
for ((b = 0; b < N_BATCHES; b++)); do
    BATCH_TASKS=()
    for ((i = b; i < ${#ALL_TASKS[@]}; i += N_BATCHES)); do
        BATCH_TASKS+=("${ALL_TASKS[$i]}")
    done
    if [ "${#BATCH_TASKS[@]}" -eq 0 ]; then
        continue
    fi
    ONLY=$(IFS=,; echo "${BATCH_TASKS[*]}")
    "$PYTHON" run_heterogeneity_v2.py --output-dir "$OUTPUT_DIR" --n-cores 1 --only "$ONLY" \
        > "${LOG_DIR}/batch_${b}.out" 2> "${LOG_DIR}/batch_${b}.err" &
    PIDS+=($!)
    echo "  started batch ${b} (pid $!, ${#BATCH_TASKS[@]} tasks)"
done

echo "[$(date)] Waiting for all ${#PIDS[@]} batches to finish..."
JOB_START=$(date +%s)
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=1
    fi
done
JOB_END=$(date +%s)

echo "[$(date)] All batches finished in $((JOB_END - JOB_START))s wall-clock."
N_COMPUTED=$(cat "${LOG_DIR}"/batch_*.out 2>/dev/null | grep -c ": computed " || true)
N_SKIPPED=$(cat "${LOG_DIR}"/batch_*.out 2>/dev/null | grep -c ": skipped " || true)
echo "[$(date)] Tasks computed: ${N_COMPUTED}, skipped (already done): ${N_SKIPPED}, total: ${#ALL_TASKS[@]}"

if [ "$FAILED" -ne 0 ]; then
    echo "[$(date)] One or more batches failed -- check ${LOG_DIR}/batch_*.err"
    exit 1
fi

echo "[$(date)] Full heterogeneity_vs_regulation rerun finished. Next: run aggregate_results.py to build the CSVs plot_v2.ipynb reads."
