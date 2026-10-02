# Source this before running anything from clean_code.  Environment: twinfer-code (call its python directly, no conda activate).
export TWINFER_PROJECT_ROOT="${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}"  # original tree holding simulation_data/, input_data/, analysis_data/, code/Beeline ...
export TWINFER_DATA_ROOT="${TWINFER_DATA_ROOT:-$TWINFER_PROJECT_ROOT/analysis_data}"            # twinfer.utils.paths.get_data_root() semantics = analysis_data
export TWINFER_CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TWINFER_REPO_ROOT="$TWINFER_CODE_ROOT"                                                    # clean_code/ is the repo root (package/pyproject.toml lives here)
export TWINFER_BEELINE_PATH="${TWINFER_BEELINE_PATH:-$TWINFER_PROJECT_ROOT/code/Beeline}"       # full Beeline install (clean_code only vendors used fragments)
export TWINFER_BOOLODE_PATH="${TWINFER_BOOLODE_PATH:-$TWINFER_PROJECT_ROOT/code/BoolODE}"
export TWINFER_PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
export PYTHONPATH="$TWINFER_CODE_ROOT/package:$TWINFER_CODE_ROOT:${PYTHONPATH:-}"
export BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"   # separate env for Beeline/BoolODE algorithm runs
export TWINFER_LARRY_DATASET="${TWINFER_LARRY_DATASET:-/scratch/gzu5140/ka_twinfer/larry_dataset}"   # LARRY preprocessing working dir (scratch is purged: point this at shared storage to keep results)
