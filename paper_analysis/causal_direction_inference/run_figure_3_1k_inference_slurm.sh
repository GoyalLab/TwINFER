#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem 32GB
#SBATCH -t 4:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/figure_3_1k_run/logs/slurm_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/figure_3_1k_run/logs/slurm_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Runs run_causal_direction_v2.py's direction-inference (Part A only, no
# cascade) against the pre-generated simulation_data/figure_3_1k/ data
# (1000 replicates each across A_rep_B, A_and_B_both_repress [folder
# A_rep_B_B_rep_A], A_rep_B_B_to_A, A_and_B [folder A_to_B_B_to_A] --
# A_to_B_B_to_A's 1377 duplicate/stale files from a prior rerun were
# deleted, keeping the latest-mtime file per rep_id; collect_tasks() also
# dedupes by rep_id now so re-adding stale files can't reintroduce this),
# then aggregate_causal_direction_results.py to build zscores.csv +
# correlation-matrix CSVs. run_main_matrices() also now calls
# run_forced_zscores(), an unconditional (ungated) reproduction of
# infer.py's Steps 1-4 for gene_1-gene_2, so zscores.csv's forced_*
# columns are always populated even when the gated pipeline's step*_z
# columns are blank (a step was skipped by an earlier step's routing).
# Same batch-of-background-processes pattern as run_causal_direction_v2_slurm.sh:
# n_cores=1 per task, real parallelism from many single-core batches
# running concurrently via SLURM_CPUS_PER_TASK. Single-task measured time
# ~20-30s (up from ~10s pre-forced-zscores) -> 4000 tasks / 20 workers
# ~= 1.3-1.5h; 4h budget leaves large margin.

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/causal_direction_inference   # [2026-09-30 replaced by env.sh variable]
# SCRIPT_DIR=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/causal_direction_inference   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR="${TWINFER_CODE_ROOT}/paper_analysis/causal_direction_inference"
# SIM_FOLDER=/home/gzu5140/TwINFER_KA/simulation_data/figure_3_1k   # [2026-09-30 replaced by env.sh variable]
SIM_FOLDER=${TWINFER_PROJECT_ROOT}/simulation_data/figure_3_1k
# OUTPUT_DIR=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/figure_3_1k_run/results   # [2026-09-30 replaced by env.sh variable]
OUTPUT_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/causal_direction_inference/figure_3_1k_run/results
# LOG_DIR=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/figure_3_1k_run/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/causal_direction_inference/figure_3_1k_run/logs
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"

N_BATCHES=${SLURM_CPUS_PER_TASK:-20}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "[$(date)] Listing all tasks..."
mapfile -t ALL_TASKS < <("$PYTHON" run_causal_direction_v2.py --sim-folder "$SIM_FOLDER" --list-tasks)
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
    "$PYTHON" run_causal_direction_v2.py --sim-folder "$SIM_FOLDER" --output-dir "$OUTPUT_DIR" --n-cores 1 --only "$ONLY" \
        > "${LOG_DIR}/batch_${b}.out" 2> "${LOG_DIR}/batch_${b}.err" &
    PIDS+=($!)
    echo "  started batch ${b} (pid $!, ${#BATCH_TASKS[@]} tasks)"
done

echo "[$(date)] Waiting for all ${#PIDS[@]} processes to finish..."
JOB_START=$(date +%s)
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=1
    fi
done
JOB_END=$(date +%s)

echo "[$(date)] All processes finished in $((JOB_END - JOB_START))s wall-clock."
N_COMPUTED=$(cat "${LOG_DIR}"/batch_*.out 2>/dev/null | grep -c ": computed " || true)
N_SKIPPED=$(cat "${LOG_DIR}"/batch_*.out 2>/dev/null | grep -c ": skipped " || true)
echo "[$(date)] Tasks computed: ${N_COMPUTED}, skipped: ${N_SKIPPED}, total: ${#ALL_TASKS[@]}"

if [ "$FAILED" -ne 0 ]; then
    echo "[$(date)] One or more processes failed -- check ${LOG_DIR}/*.err"
    exit 1
fi

echo "[$(date)] Aggregating results..."
"$PYTHON" aggregate_causal_direction_results.py --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/aggregated"

echo "[$(date)] figure_3_1k inference run finished."
