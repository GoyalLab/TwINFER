#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=mich_orig
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Figure 5 preprocessing, Michaels T1 / T2 with yscher's ORIGINAL scripts (copied here): mich_qc.py (QC of the GEO matrices: mito <= 20 %, >= 200 genes -> michaels_T1_qc.h5ad, michaels_T2_qc.h5ad,
# rule in qc.py) and michaels_barcodes.py (TLB amplicon FASTQs -> lineage barcodes -> clone graph -> cell_to_clone.csv). Output: <REBUILD>/Michaels_T1T2/{processed,clones}.
# mich_qc.py reproduces her QC files exactly (checked 2026-10-23). The clone ids of michaels_barcodes.py depend on PYTHONHASHSEED (step 8), so a rerun gives about the same clones, not the same ids
# (see michaels_step8_seeds.py); the labels of the figure are yscher's cell_to_clone.csv.   Usage: sbatch run_michaels_barcodes_and_qc.sbatch
set -uo pipefail
ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
FIG=$ROOT/final_figure_data/real_data_analysis_and_melanoma
RAW=${FIG5_RAW_DATA:-$FIG/raw_data}                                  # links to the raw data (see raw_data/)
FINAL=$FIG/final_data                                                # links to the final objects
REBUILD=${FIG5_REBUILD_DIR:-$FIG/analysis_data/preprocessing_rebuild}  # rebuilt files go here, never into final_data
CODE=$ROOT/code/TwINFER/paper_analysis/real_data_analysis_and_melanoma/preprocessing_code
PY=${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}   # conda activate does not work in batch jobs: call the environment python directly
export TWINFER_PROJECT_ROOT=$ROOT
mkdir -p "$REBUILD/Michaels_T1T2"
cd "$CODE/Michaels_T1T2"
$PY mich_qc.py > "$REBUILD/Michaels_T1T2/mich_qc.log" 2>&1
$RAW/external_tools/mamba_python3.10 michaels_barcodes.py > "$REBUILD/Michaels_T1T2/michaels_barcodes.log" 2>&1      # python 3.10 with networkx, as for her run
