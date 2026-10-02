#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --mem 64GB
#SBATCH -t 4:00:00
#SBATCH --job-name=infer_larry
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/%x_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/%x_%j.err
# Generic LARRY TwINFER inference launcher (2026-09-30): replaces the 15 run_infer_{correlation,detection,variability,allpairs*,yscher_single*,correlation_high,allpairs20_all9}.sh
# launchers (originals in _archive/paper_analysis/larry_hematopoiesis_validation/superseded_infer_launchers/). Every run calls run_infer_correlation_high.py with environment settings.
# Usage:  source clean_code/env.sh
#         sbatch --job-name=infer_<run> [resources from the run file] run_infer_larry.sh <run>      settings: infer_runs/<run>.env
#           MODE=loop    one job runs every gene set in GENE_SETS one after the other
#           MODE=array   one gene set per array task (sbatch --array=0-8): GENE_SETS[$SLURM_ARRAY_TASK_ID]
#           MODE=env     GENE_SET must be given:  sbatch --export=ALL,GENE_SET=correlation_mid run_infer_larry.sh <run>
#           MODE=default no GENE_SET (the driver's default, correlation_high)
#         bash run_infer_larry.sh <run> --dry-run     prints the commands without running them.
# The SBATCH header above is the common default; each .env lists the original job's resources on its second line (sbatch command-line options override the header).
set -euo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
RUN="${1:?usage: run_infer_larry.sh <run> [--dry-run]}"; DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
HERE_DIR="${TWINFER_CODE_ROOT}/paper_analysis/larry_hematopoiesis_validation/preprocessing"
ENVFILE="${HERE_DIR}/infer_runs/${RUN}.env"
[ -f "$ENVFILE" ] || { echo "no such run: $ENVFILE" >&2; exit 2; }
unset MODE GENE_SETS ALL_PAIRS OUT_SUFFIX N_CORES INPUT_DIR
source "$ENVFILE"
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
cd "${TWINFER_PROJECT_ROOT}"
[ -n "${ALL_PAIRS+x}" ] && export ALL_PAIRS; [ -n "${OUT_SUFFIX+x}" ] && export OUT_SUFFIX
[ -n "${N_CORES+x}" ] && export N_CORES;     [ -n "${INPUT_DIR+x}" ] && export INPUT_DIR
run_one() {   # $1 = gene set or "" (driver default)
    if [ -n "$1" ]; then export GENE_SET="$1"; fi
    echo "[$(date)] TwINFER LARRY run=$RUN gene_set=${GENE_SET:-<driver default>} ALL_PAIRS=${ALL_PAIRS:-0} OUT_SUFFIX=${OUT_SUFFIX:-} N_CORES=${N_CORES:-8} INPUT_DIR=${INPUT_DIR:-twinfer_input}" >&2
    [ "$DRY" = 1 ] || "$PYTHON" "${HERE_DIR}/run_infer_correlation_high.py"
    echo "[$(date)] Done (${GENE_SET:-default})" >&2
}
case "$MODE" in
  loop)    for GS in $GENE_SETS; do run_one "$GS"; done ;;
  array)   read -ra ARR <<< "$GENE_SETS"; run_one "${ARR[$SLURM_ARRAY_TASK_ID]}" ;;
  env)     : "${GENE_SET:?GENE_SET must be set, e.g. sbatch --export=ALL,GENE_SET=correlation_mid run_infer_larry.sh $RUN}"; run_one "" ;;
  default) run_one "" ;;
  *) echo "bad MODE=$MODE in $ENVFILE" >&2; exit 2 ;;
esac
