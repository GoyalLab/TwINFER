#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/slurm_run/logs/slurm_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/slurm_run/logs/slurm_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Full Parts A+B run: 120 direction-inference replicates (6 conditions x 20
# reps, run_causal_direction_v2.py) + the 1 cascade replicate
# (run_cascade_v2.py), then aggregate_causal_direction_results.py to build
# the CSVs plot_v2.ipynb reads. Same batch-of-background-processes pattern
# validated for heterogeneity_vs_regulation (job 5228767/5228968):
# n_cores=1 per task, all real parallelism from running many single-core
# batches concurrently via SLURM_CPUS_PER_TASK.

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/causal_direction_inference   # [2026-09-30 replaced by env.sh variable]
# SCRIPT_DIR=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/causal_direction_inference   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR="${TWINFER_CODE_ROOT}/paper_analysis/causal_direction_inference"
# OUTPUT_DIR=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/slurm_run/results   # [2026-09-30 replaced by env.sh variable]
OUTPUT_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/causal_direction_inference/slurm_run/results
# CASCADE_OUTPUT_DIR=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/slurm_run/cascade_results   # [2026-09-30 replaced by env.sh variable]
CASCADE_OUTPUT_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/causal_direction_inference/slurm_run/cascade_results
# LOG_DIR=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/slurm_run/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/causal_direction_inference/slurm_run/logs
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"

N_BATCHES=${SLURM_CPUS_PER_TASK:-20}

mkdir -p "$OUTPUT_DIR" "$CASCADE_OUTPUT_DIR" "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "[$(date)] Listing all Part A tasks..."
mapfile -t ALL_TASKS < <("$PYTHON" run_causal_direction_v2.py --list-tasks)
echo "[$(date)] Got ${#ALL_TASKS[@]} Part A task(s); splitting into ${N_BATCHES} batch(es)."

echo "[$(date)] Launching Part A parallel batches..."
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
    "$PYTHON" run_causal_direction_v2.py --output-dir "$OUTPUT_DIR" --n-cores 1 --only "$ONLY" \
        > "${LOG_DIR}/partA_batch_${b}.out" 2> "${LOG_DIR}/partA_batch_${b}.err" &
    PIDS+=($!)
    echo "  started Part A batch ${b} (pid $!, ${#BATCH_TASKS[@]} tasks)"
done

"$PYTHON" run_cascade_v2.py --output-dir "$CASCADE_OUTPUT_DIR" --n-cores 1 \
    > "${LOG_DIR}/partB_cascade.out" 2> "${LOG_DIR}/partB_cascade.err" &
PIDS+=($!)
echo "  started Part B cascade (pid $!)"

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
N_COMPUTED=$(cat "${LOG_DIR}"/partA_batch_*.out 2>/dev/null | grep -c ": computed " || true)
N_SKIPPED=$(cat "${LOG_DIR}"/partA_batch_*.out 2>/dev/null | grep -c ": skipped " || true)
echo "[$(date)] Part A tasks computed: ${N_COMPUTED}, skipped: ${N_SKIPPED}, total: ${#ALL_TASKS[@]}"

if [ "$FAILED" -ne 0 ]; then
    echo "[$(date)] One or more processes failed -- check ${LOG_DIR}/*.err"
    exit 1
fi

echo "[$(date)] Aggregating Part A results..."
"$PYTHON" aggregate_causal_direction_results.py --input-dir "$OUTPUT_DIR"

echo "[$(date)] Scoring Part B cascade results..."
"$PYTHON" score_cascade_results.py --input-dir "$CASCADE_OUTPUT_DIR" \
    --output-csv "${CASCADE_OUTPUT_DIR}/metrics_TwINFER.csv"

echo "[$(date)] causal_direction_inference Parts A+B run finished."
