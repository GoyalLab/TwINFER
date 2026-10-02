#!/bin/bash
#SBATCH --account=p32655
#SBATCH --job-name=twinscore_larry
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
# Generic LARRY TwinScore-supplement run (replaces 15 run_twinscore_{correlation,variability}_{low,mid,high}[_v2|_par] / run_twinscore_supplement_mid scripts, see _archive/paper_analysis/larry_hematopoiesis_validation/superseded_sbatch/).
# Usage:  source clean_code/env.sh ; sbatch --job-name=twinscore_<name> run_twinscore_larry.sbatch <run-name>      (settings: twinscore_runs/<run-name>.env: GENE_SETS [, SUMMARY_SUFFIX])
#         bash run_twinscore_larry.sbatch <run-name> --dry-run   prints the command without running it. sbatch exports the environment, so source env.sh before submitting.
set -eo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
RUN="${1:?usage: run_twinscore_larry.sbatch <run-name> [--dry-run]}"; DRY=0; [ "$2" = "--dry-run" ] && DRY=1
HERE_DIR="${TWINFER_CODE_ROOT}/paper_analysis/larry_hematopoiesis_validation"
ENVFILE="${HERE_DIR}/twinscore_runs/${RUN}.env"
[ -f "$ENVFILE" ] || { echo "no such run: $ENVFILE" >&2; exit 2; }
unset GENE_SETS SUMMARY_SUFFIX
source "$ENVFILE"
cd "$HERE_DIR"
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
export GENE_SETS
[ -n "${SUMMARY_SUFFIX+x}" ] && export SUMMARY_SUFFIX
echo "+ GENE_SETS=$GENE_SETS${SUMMARY_SUFFIX+ SUMMARY_SUFFIX=$SUMMARY_SUFFIX} $PYTHON apply_twinscore_supplement_larry.py" >&2
[ "$DRY" = 1 ] || "$PYTHON" apply_twinscore_supplement_larry.py
