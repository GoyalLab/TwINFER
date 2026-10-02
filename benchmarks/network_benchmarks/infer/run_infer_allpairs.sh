#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
# Generic ALL_PAIRS TwINFER inference launcher (2026-10-01): replaces run_mixed_network_sweep_allpairs(.sh|_t1_10.sh), run_network_sweep_final_allpairs(...), run_real_network_allpairs(...),
# run_e13_pos100_allpairs_t1_10.sh (originals in _archive/benchmarks/network_benchmarks/infer/superseded_launchers/). Settings per run: runs/<run>.env.
# Usage:  source clean_code/env.sh
#         sbatch <SBATCH_HINT from runs/<run>.env> --job-name=<run> --output=<logdir>/<run>_%A_%a.out --error=<logdir>/<run>_%A_%a.err benchmarks/network_benchmarks/infer/run_infer_allpairs.sh <run>
#         bash run_infer_allpairs.sh <run> --dry-run      prints the command without running it (SLURM_ARRAY_TASK_ID picks the chunk if set)
#   MODE=single | array_chunk (task i runs rows [i*CHUNK, i*CHUNK+CHUNK) via --start/--count); SCRIPT; ARGS; T1_VARIANT (exported as TWINFER_T1_VARIANT when set).
# The per-run #SBATCH resources/array/log paths cannot be variables, so pass them on the sbatch command line (SBATCH_HINT records what the original script used).
set -euo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
RUN="${1:?usage: run_infer_allpairs.sh <run> [--dry-run]}"; DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
HERE_DIR="${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/infer"
ENVFILE="${HERE_DIR}/runs/${RUN}.env"
[ -f "$ENVFILE" ] || { echo "no such run: $ENVFILE" >&2; exit 2; }
MODE=single; SCRIPT=; ARGS=(); CHUNK=; T1_VARIANT=; SBATCH_HINT=
source "$ENVFILE"
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
cd "$HERE_DIR"
[ -n "$T1_VARIANT" ] && export TWINFER_T1_VARIANT="$T1_VARIANT"
cmd=("$PYTHON" "$SCRIPT" "${ARGS[@]}")
if [ "$MODE" = array_chunk ]; then
    : "${SLURM_ARRAY_TASK_ID:?MODE=array_chunk needs SLURM_ARRAY_TASK_ID (use --array)}"
    START=$((SLURM_ARRAY_TASK_ID * CHUNK))
    echo "[$(date)] run=$RUN array task $SLURM_ARRAY_TASK_ID: tasks[$START:$((START+CHUNK))]" >&2
    cmd+=(--start "$START" --count "$CHUNK")
else
    echo "[$(date)] run=$RUN" >&2
fi
echo "+ ${T1_VARIANT:+TWINFER_T1_VARIANT=$T1_VARIANT }${cmd[*]}" >&2
[ "$DRY" = 1 ] || "${cmd[@]}"
echo "[$(date)] Done" >&2
