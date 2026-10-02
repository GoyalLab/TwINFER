#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --mem 48GB
#SBATCH -t 4:00:00
#SBATCH --job-name=het_v2_full
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_logs/slurm_%x_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_logs/slurm_%x_%j.err
# Full production run of all ~4002 heterogeneity_vs_regulation Figure-2 tasks (A_to_B, A_B, A_to_B_2_states, A_B_2_states) via run_heterogeneity_v2.py.
# 2026-09-30: merges run_heterogeneity_v2_slurm_full.sh and run_heterogeneity_v2_slurm_full_rerun.sh (originals in _archive/paper_analysis/heterogeneity_vs_regulation/superseded_launchers/):
# the two differed only in the output directory (slurm_full_run vs slurm_full_run_v2).
# Usage:  source clean_code/env.sh ; mkdir -p $TWINFER_PROJECT_ROOT/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_logs
#         sbatch run_heterogeneity_v2_slurm.sh <run>            settings: heterogeneity_runs/<run>.env  (RUN_DIR = subfolder of heterogeneity_vs_regulation_1k_v2 for results/logs)
#           full         -> slurm_full_run     (first production run)
#           full_rerun   -> slurm_full_run_v2  (fresh directory: process_replicate() now also stores Step 2 diagnostics; the script skips tasks whose output JSON exists, so reusing
#                                                the old directory would silently keep the old, incomplete records)
#         bash run_heterogeneity_v2_slurm.sh <run> --dry-run      lists what would run (no python).
# Same batch-of-background-processes pattern validated by the timing test (job 5228767: 40 tasks / 20 cores in 130s): ~4002/40 =~ 101 tasks per batch x ~65s/task =~ 1.8-2h; safe to
# resubmit if it times out (existing outputs are skipped).
set -euo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
RUN="${1:?usage: run_heterogeneity_v2_slurm.sh <run> [--dry-run]}"; DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
SCRIPT_DIR="${TWINFER_CODE_ROOT}/paper_analysis/heterogeneity_vs_regulation"
ENVFILE="${SCRIPT_DIR}/heterogeneity_runs/${RUN}.env"
[ -f "$ENVFILE" ] || { echo "no such run: $ENVFILE" >&2; exit 2; }
unset RUN_DIR; source "$ENVFILE"

OUTPUT_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/${RUN_DIR}/results
# LOG_DIR=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/slurm_full_run/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k_v2/${RUN_DIR}/logs
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"

N_BATCHES=${SLURM_CPUS_PER_TASK:-40}

if [ "$DRY" = 1 ]; then echo "run=$RUN  results -> $OUTPUT_DIR  logs -> $LOG_DIR  (then: $PYTHON run_heterogeneity_v2.py --list-tasks; N_BATCHES=$N_BATCHES background batches of: run_heterogeneity_v2.py --output-dir ... --n-cores 1 --only ...)"; exit 0; fi
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

echo "[$(date)] Full heterogeneity_vs_regulation run finished. Next: run aggregate_results.py to build the CSVs plot_v2.ipynb reads."
