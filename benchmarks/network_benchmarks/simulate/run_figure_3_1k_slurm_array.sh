#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=50
#SBATCH --mem 96GB
#SBATCH -t 24:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/simulation_data/figure_3_1k/logs/slurm_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/simulation_data/figure_3_1k/logs/slurm_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Generates the 4 figure_3 networks not already in figure_2 (A_rep_B,
# A_and_B_both_repress, A_rep_B_B_to_A, A_and_B), 1000 replicates apiece
# (4000 total), via run_figure_3_1k_simulation.py. 5 array tasks x 50
# concurrent single-threaded replicates each = 250-way total concurrency.
# n_cores=1 per replicate throughout -- measured: wall-clock/replicate is
# best with more cores (132.9s at 20 cores vs 1872.7s at 1 core), but
# *throughput per core* for a fixed total budget is best at n_cores=1
# (parallel efficiency drops from ~77% at 4 cores to worse at 20), so many
# concurrent single-threaded replicates beats fewer multi-threaded ones.
# Estimated: 4000 replicates / 250-way concurrency x ~1873s/replicate =~
# 8.3h. Resumable -- run_figure_3_1k_simulation.py skips any
# condition/rep_id whose output file already exists, safe to resubmit if
# a task array member times out.

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/TwINFER/scripts_simulation_for_figures   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR="${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/simulate"
# LOG_DIR=/home/gzu5140/TwINFER_KA/simulation_data/figure_3_1k/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/simulation_data/figure_3_1k/logs
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"

N_ARRAY_TASKS=2
ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
N_BATCHES=${SLURM_CPUS_PER_TASK:-50}

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "[$(date)] Array task ${ARRAY_IDX}/${N_ARRAY_TASKS}: listing all tasks..."
mapfile -t ALL_TASKS < <("$PYTHON" run_figure_3_1k_simulation.py --list-tasks)
echo "[$(date)] Got ${#ALL_TASKS[@]} total tasks across all array members."

# This array task's shard: every N_ARRAY_TASKS-th task, offset by ARRAY_IDX.
MY_TASKS=()
for ((i = ARRAY_IDX; i < ${#ALL_TASKS[@]}; i += N_ARRAY_TASKS)); do
    MY_TASKS+=("${ALL_TASKS[$i]}")
done
echo "[$(date)] Array task ${ARRAY_IDX} owns ${#MY_TASKS[@]} tasks; splitting into ${N_BATCHES} batch(es)."

echo "[$(date)] Launching parallel batches..."
declare -a PIDS=()
for ((b = 0; b < N_BATCHES; b++)); do
    BATCH_TASKS=()
    for ((i = b; i < ${#MY_TASKS[@]}; i += N_BATCHES)); do
        BATCH_TASKS+=("${MY_TASKS[$i]}")
    done
    if [ "${#BATCH_TASKS[@]}" -eq 0 ]; then
        continue
    fi
    ONLY=$(IFS=,; echo "${BATCH_TASKS[*]}")
    "$PYTHON" run_figure_3_1k_simulation.py --n-cores 1 --only "$ONLY" \
        > "${LOG_DIR}/array${ARRAY_IDX}_batch${b}.out" 2> "${LOG_DIR}/array${ARRAY_IDX}_batch${b}.err" &
    PIDS+=($!)
done
echo "[$(date)] Started ${#PIDS[@]} background batch(es) for array task ${ARRAY_IDX}."

echo "[$(date)] Waiting for all ${#PIDS[@]} batches to finish..."
JOB_START=$(date +%s)
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=1
    fi
done
JOB_END=$(date +%s)

echo "[$(date)] Array task ${ARRAY_IDX} finished in $((JOB_END - JOB_START))s wall-clock."
N_COMPUTED=$(cat "${LOG_DIR}"/array${ARRAY_IDX}_batch*.out 2>/dev/null | grep -c ": computed " || true)
N_SKIPPED=$(cat "${LOG_DIR}"/array${ARRAY_IDX}_batch*.out 2>/dev/null | grep -c ": skipped " || true)
echo "[$(date)] Array task ${ARRAY_IDX}: computed ${N_COMPUTED}, skipped ${N_SKIPPED}, owned ${#MY_TASKS[@]}."

if [ "$FAILED" -ne 0 ]; then
    echo "[$(date)] One or more batches failed -- check ${LOG_DIR}/array${ARRAY_IDX}_batch*.err"
    exit 1
fi

echo "[$(date)] Array task ${ARRAY_IDX} done."
