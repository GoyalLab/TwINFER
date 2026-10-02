#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
#SBATCH --time=4:00:00
#SBATCH --job-name=drift_grid
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/drift_inference/logs/%x_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/drift_inference/logs/%x_%A_%a.err
# Generic drift z-score / inference launcher (2026-09-30): replaces run_6scenario_zscores.sh, run_8scenario_zscores.sh, run_8scenario_drift_only.sh,
# run_abs_drift_zscores.sh, run_abs_drift_zscores_t10.sh, run_infer_drift_replicates.sh, run_infer_drift_replicates_10k.sh
# (originals in _archive/simulations/drift_multiple_state/superseded_launchers/). Conda activation is not used: TWINFER_PYTHON is called directly.
# Usage:  source clean_code/env.sh
#         sbatch --job-name=<run> [--array=0-N-1 and the resources listed in the run file] run_drift_grid.sh <run>         settings: drift_runs/<run>.env
#         bash run_drift_grid.sh <run> --dry-run          prints the commands without running them (SLURM_ARRAY_TASK_ID picks one task if set).
#   MODE=single | array (task = $SLURM_ARRAY_TASK_ID); SCRIPT, PY_FLAGS, ARGS (@OUT@ = the task's output dir), EXPORTS (all tasks), TASK_EXPORTS[i], TASK_OUT[i], MKDIRS.
set -euo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
RUN="${1:?usage: run_drift_grid.sh <run> [--dry-run]}"; DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
HERE_DIR="${TWINFER_CODE_ROOT}/simulations/drift_multiple_state"
ENVFILE="${HERE_DIR}/drift_runs/${RUN}.env"
[ -f "$ENVFILE" ] || { echo "no such run: $ENVFILE" >&2; exit 2; }
MODE=single; SCRIPT=; PY_FLAGS=(); ARGS=(); EXPORTS=(); TASK_EXPORTS=(); TASK_OUT=(); MKDIRS=()
source "$ENVFILE"
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
cd "$HERE_DIR"
for e in "${EXPORTS[@]+"${EXPORTS[@]}"}"; do export "$e"; done
run_task() {   # $1 = task index
    local i=$1 out="" a=() x
    [ ${#TASK_OUT[@]} -gt "$i" ] && out="${TASK_OUT[$i]}"
    if [ ${#TASK_EXPORTS[@]} -gt "$i" ]; then for e in ${TASK_EXPORTS[$i]}; do export "$e"; done; fi
    for x in "${ARGS[@]+"${ARGS[@]}"}"; do a+=("${x//@OUT@/$out}"); done
    if [ "$DRY" != 1 ]; then for d in "${MKDIRS[@]+"${MKDIRS[@]}"}"; do mkdir -p "${d//@OUT@/$out}"; done; fi
    echo "[$(date)] run=$RUN task $i: ${TASK_EXPORTS[$i]:-} $SCRIPT ${a[*]}" >&2
    [ "$DRY" = 1 ] || "$PYTHON" "${PY_FLAGS[@]+"${PY_FLAGS[@]}"}" "$SCRIPT" "${a[@]}"
    echo "[$(date)] DONE run=$RUN task $i" >&2
}
case "$MODE" in
  single) run_task 0 ;;
  array)  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then run_task "$SLURM_ARRAY_TASK_ID"; elif [ "$DRY" = 1 ]; then for ((i=0;i<${#TASK_OUT[@]};i++)); do run_task $i; done; else echo "MODE=array needs SLURM_ARRAY_TASK_ID" >&2; exit 2; fi ;;
  *) echo "bad MODE=$MODE" >&2; exit 2 ;;
esac
