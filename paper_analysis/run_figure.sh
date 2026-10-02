#!/bin/bash
# Run the stages of one figure folder (written 2026-10-01, REORG_CHECKLIST 2.11). Nothing here submits SLURM jobs.
# Usage:  source clean_code/env.sh
#         bash paper_analysis/run_figure.sh <folder> [simulate|analyze|plot|all] [--variant v2] [--dry-run]
#   <folder>   a folder of paper_analysis/ (see figures_manifest.yaml), e.g. heterogeneity_vs_regulation
#   stages     simulate -> simulate.py | preprocess.py ; analyze -> analysis[_v2].ipynb ; plot -> plot[_v2].ipynb   (default: all, in that order)
#   --variant v2   use analysis_v2.ipynb / plot_v2.ipynb (the newer z-score version) where the folder has them
#   --dry-run      print what would run
# A stage whose file the folder does not have is skipped with a message. Notebooks are executed with nbconvert (twinfer-code python); the executed
# copies are written to clean_data/paper_analysis/<folder>/executed_notebooks/ (override: TWINFER_FIGURE_OUT), the source notebooks stay untouched.
# Scripts run from inside the folder. Simulation stages can take hours: use --dry-run first. Settings live in the scripts/notebooks themselves.
set -euo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh first}"; : "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh first}"
PAPER_DIR="${TWINFER_PAPER_DIR:-${TWINFER_CODE_ROOT}/paper_analysis}"      # override only for tests
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
FOLDER="${1:?usage: run_figure.sh <folder> [simulate|analyze|plot|all] [--variant v2] [--dry-run]}"; shift
STAGE=all; VARIANT=""; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  simulate|analyze|plot|all) STAGE=$1;; --variant) VARIANT="_${2:?--variant needs a value}"; shift;; --dry-run) DRY=1;;
  *) echo "unknown argument: $1" >&2; exit 2;; esac; shift; done
DIR="${PAPER_DIR}/${FOLDER}"; [ -d "$DIR" ] || { echo "no such folder: $DIR" >&2; exit 2; }
OUT="${TWINFER_FIGURE_OUT:-${TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/${FOLDER}/executed_notebooks}"
say() { echo "[run_figure ${FOLDER}] $*" >&2; }
run_py() { say "python $1"; [ "$DRY" = 1 ] || (cd "$DIR" && "$PYTHON" "$1"); }
run_nb() { say "execute $1 -> $OUT"; [ "$DRY" = 1 ] || { mkdir -p "$OUT"; (cd "$DIR" && "$PYTHON" -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir "$OUT" "$1"); }; }
want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
if want simulate; then
  if   [ -f "$DIR/simulate.py" ];   then run_py simulate.py
  elif [ -f "$DIR/preprocess.py" ]; then run_py preprocess.py
  else say "simulate: skipped (no simulate.py / preprocess.py)"; fi
fi
if want analyze; then
  if   [ -f "$DIR/analysis${VARIANT}.ipynb" ]; then run_nb "analysis${VARIANT}.ipynb"
  else say "analyze: skipped (no analysis${VARIANT}.ipynb)"; fi
fi
if want plot; then
  if   [ -f "$DIR/plot${VARIANT}.ipynb" ]; then run_nb "plot${VARIANT}.ipynb"
  else say "plot: skipped (no plot${VARIANT}.ipynb)"; fi
fi
say "done"
