#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=celltag_67
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Figure 5 preprocessing, CellTag3 steps 6-7 only (sparse-clone repair + repaired schema) in the work tree <REBUILD>/CellTag3/mirror that run_rebuild_celltag.sbatch (steps 1-5) has built.
# yscher's scripts and interpreter; igraph (< 0.10, the authors' integer-named vertex column), shap, upsetplot and adjustText installed with  pip install --user  into that python; cassiopeia
# stays stubbed exactly as yscher's script does.   Usage: sbatch run_rebuild_celltag_steps67.sbatch
set -uo pipefail
ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
FIG=$ROOT/final_figure_data/real_data_analysis_and_melanoma
RAW=${FIG5_RAW_DATA:-$FIG/raw_data}                                  # links to the raw data (see raw_data/)
FINAL=$FIG/final_data                                                # links to the final objects
REBUILD=${FIG5_REBUILD_DIR:-$FIG/analysis_data/preprocessing_rebuild}  # rebuilt files go here, never into final_data
CODE=$ROOT/code/TwINFER/paper_analysis/real_data_analysis_and_melanoma/preprocessing_code
PY_TWINFER=${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}   # conda activate does not work in batch jobs: call the environment python directly
export TWINFER_PROJECT_ROOT=$ROOT
W=$REBUILD/CellTag3; M=$W/mirror
PY=$RAW/external_tools/mamba_python3.10
CMP="$PY_TWINFER $CODE/h5ad_identity.py"
rsync -a --exclude __pycache__ --exclude '*.sbatch' "$CODE/CellTag3/" $M/helpers/      # refresh the copy of the scripts
cp "$CODE/fig5_paths.py" $M/fig5_paths.py
cd $M; export MPLBACKEND=Agg PYTHONUNBUFFERED=1
step(){ echo "=== $1 $(date)"; shift; "$@" > $W/step_$(date +%s).log 2>&1; echo "exit $?"; }
D=$M/datasets/celltag; YD=$RAW/yscher_celltag_datasets; YE=$RAW/yscher_celltag_clone_exports; E=exports/legacy_reproductions/pairing/celltag_clones     # her files = reference; $M/$E = the rebuilt ones
step 6_repaired_clones  $PY helpers/celltag_clones_repaired.py
cmp $M/$E/clone_table_repaired.csv "$YE/clone_table_repaired.csv" && echo "clone_table_repaired.csv byte-identical" || echo "clone_table_repaired.csv DIFFERS"
step 7_repaired_schema  $PY helpers/build_celltag_repaired_schema.py
for f in celltag_clones_repaired_nopad.h5ad celltag_clones_repaired_nopad_counts.h5ad; do $CMP $D/$f "$YD/$f" $W/cmp_7_$f.json | tail -2; done
md5sum $D/celltag_clones_repaired_nopad.h5ad $FINAL/cellTag3_data/celltag_clones_repaired_nopad.h5ad
echo "done $(date)"
