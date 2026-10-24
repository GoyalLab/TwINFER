#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_celltag
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# Figure 5 preprocessing, CellTag3 (final file: celltag_clones_repaired_nopad.h5ad): the 7-step chain run with yscher's scripts (copied here) and her interpreter (python 3.10 of the shared
# mamba, raw_data/external_tools) from the GEO Cell Ranger matrices + all_samples.csv + the authors' master clone table. The scripts work relative to helpers/../datasets and ../exports, so the
# launcher builds a work tree <REBUILD>/CellTag3/mirror (copy of this folder as helpers/, empty datasets/ and exports/) and every intermediate is compared with yscher's file of the same name
# (h5ad: element-wise; csv: byte). Last line: md5 of the rebuilt final file vs final_data/cellTag3_data. final_data is never touched.   Usage: sbatch run_rebuild_celltag.sbatch
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
mkdir -p $M/datasets/celltag $M/exports/legacy_reproductions/pairing/celltag_clones $M/exports/qc
rsync -a --exclude __pycache__ --exclude '*.sbatch' "$CODE/CellTag3/" $M/helpers/      # copy of the scripts (not a link)
cp "$CODE/fig5_paths.py" $M/fig5_paths.py                                              # imported by helpers/ scripts through sys.path .. = the mirror root
cd $M; export MPLBACKEND=Agg PYTHONUNBUFFERED=1
step(){ echo "=== $1 $(date)"; shift; "$@" > $W/step_$(date +%s).log 2>&1; echo "exit $?"; }
D=$M/datasets/celltag; YD=$RAW/yscher_celltag_datasets; YE=$RAW/yscher_celltag_clone_exports; E=exports/legacy_reproductions/pairing/celltag_clones     # her files = reference; $M/$E = the rebuilt ones
step 1_build_nosinglet  $PY helpers/build_nosinglet_dataset.py
step 2_finalize         $PY helpers/finalize_nosinglet.py
$CMP $D/celltag_nosinglet_counts.h5ad "$YD/celltag_nosinglet_counts.h5ad" $W/cmp_1_nosinglet.json | tail -3
step 3_qc               env CT_SOURCE=$D/celltag_nosinglet_counts.h5ad $PY helpers/celltag_apply_qc.py
$CMP $D/celltag_nosinglet_counts_mito20_min200genes.h5ad "$YD/celltag_nosinglet_counts_mito20_min200genes.h5ad" $W/cmp_3_qc.json | tail -3
step 4_clones           env CT_SOURCE=$D/celltag_nosinglet_counts_mito20_min200genes.h5ad PAIR_SUFFIX=_mito20_min200genes TAG_MIN=2 TAG_MAX=24 $PY helpers/build_corrected_clones.py
for f in clone_assignments_corrected_mito20_min200genes.csv pairs_barcode_overlap_2to24tags_mito20_min200genes.csv; do cmp $M/$E/$f "$YE/$f" && echo "$f byte-identical" || echo "$f DIFFERS"; done
step 5_published_nopad  env CT_SOURCE=$D/celltag_nosinglet_counts_mito20_min200genes.h5ad OUT_STEM=celltag_clones_published_nopad $PY helpers/build_celltag_schema_clones.py
for f in celltag_clones_published_nopad.h5ad celltag_clones_published_nopad_counts.h5ad; do $CMP $D/$f "$YD/$f" $W/cmp_5_$f.json | tail -2; done
step 6_repaired_clones  $PY helpers/celltag_clones_repaired.py
cmp $M/$E/clone_table_repaired.csv "$YE/clone_table_repaired.csv" && echo "clone_table_repaired.csv byte-identical" || echo "clone_table_repaired.csv DIFFERS"
step 7_repaired_schema  $PY helpers/build_celltag_repaired_schema.py
for f in celltag_clones_repaired_nopad.h5ad celltag_clones_repaired_nopad_counts.h5ad; do $CMP $D/$f "$YD/$f" $W/cmp_7_$f.json | tail -2; done
md5sum $D/celltag_clones_repaired_nopad.h5ad $FINAL/cellTag3_data/celltag_clones_repaired_nopad.h5ad
echo "done $(date)"
