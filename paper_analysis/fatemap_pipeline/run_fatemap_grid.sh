#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 12:00:00
#SBATCH --job-name=fatemap_grid
#SBATCH --output=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/%x_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/%x_%A_%a.err
# Generic fatemap "script x dataset x gene-set grid" launcher (2026-09-30): replaces 22 run_{infer,ab_infer,wm_ab_infer,supp_gated_bootstrap,wm_supp_gated_bootstrap,competitors}_*.sh
# launchers that differed only in script, dataset list, gene-set list and arguments (originals in _archive/paper_analysis/fatemap_pipeline/superseded_grid_launchers/).
# Usage:  source clean_code/env.sh
#         sbatch --job-name=<run> [--array=0-N-1 and the resources listed in the run file] run_fatemap_grid.sh <run>        settings: grid_runs/<run>.env
#         bash run_fatemap_grid.sh <run> --dry-run          prints the commands of ALL tasks without running them (SLURM_ARRAY_TASK_ID picks one task if set).
# Task order = dataset-major, gene-set-minor (same as the original `ID / N_GENE_SETS`, `ID % N_GENE_SETS`).
#   MODE=array  task = $SLURM_ARRAY_TASK_ID;  MODE=loop  all tasks one after the other;  MODE=single  exactly one task.
#   ARGS may contain @DATASET@ and @GENE_SET@;  PER_TASK_ARGS[i] is appended for task i (used when tasks differ only by a flag);  EXPORTS are exported first.
# The SBATCH header is the common default; the second line of each .env lists the original job's resources (sbatch command-line options override the header).
set -euo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
RUN="${1:?usage: run_fatemap_grid.sh <run> [--dry-run]}"; DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
HERE_DIR="${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"
ENVFILE="${HERE_DIR}/grid_runs/${RUN}.env"
[ -f "$ENVFILE" ] || { echo "no such run: $ENVFILE" >&2; exit 2; }
unset MODE SCRIPT DATASETS GENE_SETS ARGS PER_TASK_ARGS EXPORTS PY_FLAGS
DATASETS=(); GENE_SETS=(); ARGS=(); PER_TASK_ARGS=(); EXPORTS=(); PY_FLAGS=()
source "$ENVFILE"
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
cd "$HERE_DIR"
for e in "${EXPORTS[@]+"${EXPORTS[@]}"}"; do export "$e"; done
D=("${DATASETS[@]+"${DATASETS[@]}"}"); [ ${#D[@]} -eq 0 ] && D=("")
G=("${GENE_SETS[@]+"${GENE_SETS[@]}"}"); [ ${#G[@]} -eq 0 ] && G=("")
TD=(); TG=()
for d in "${D[@]}"; do for g in "${G[@]}"; do TD+=("$d"); TG+=("$g"); done; done
N=${#TD[@]}; [ ${#PER_TASK_ARGS[@]} -gt $N ] && N=${#PER_TASK_ARGS[@]}
run_task() {   # $1 = task index
    local i=$1 a=() x
    for x in "${ARGS[@]+"${ARGS[@]}"}"; do x="${x//@DATASET@/${TD[$i]:-}}"; x="${x//@GENE_SET@/${TG[$i]:-}}"; a+=("$x"); done
    [ ${#PER_TASK_ARGS[@]} -gt "$i" ] && [ -n "${PER_TASK_ARGS[$i]}" ] && a+=(${PER_TASK_ARGS[$i]})
    echo "[$(date)] task $i/$N run=$RUN: $SCRIPT ${a[*]}" >&2
    [ "$DRY" = 1 ] || "$PYTHON" "${PY_FLAGS[@]+"${PY_FLAGS[@]}"}" "$SCRIPT" "${a[@]}"
    echo "[$(date)] Done" >&2
}
case "$MODE" in
  array)  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then run_task "$SLURM_ARRAY_TASK_ID"; elif [ "$DRY" = 1 ]; then for ((i=0;i<N;i++)); do run_task $i; done; else echo "MODE=array needs SLURM_ARRAY_TASK_ID (sbatch --array=0-$((N-1)))" >&2; exit 2; fi ;;
  loop)   for ((i=0;i<N;i++)); do run_task $i; done ;;
  single) run_task 0 ;;
  *) echo "bad MODE=$MODE in $ENVFILE" >&2; exit 2 ;;
esac
