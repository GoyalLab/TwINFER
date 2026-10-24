#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=larry_t1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
# Figure 5 preprocessing, LARRY stage 2: (1) the dip-test + Otsu whitelist build (comparison only), (2) the build with the Weinreb et al. Table 1 whitelist (minimum UMIs: day 2 500,
# day 4 300, day 6 300) with the corrected criterion-4 rescue and no Weinreb rule (= the final LARRY file), (3) the cell-type annotation of (2), (4) comparison of (1) and (2).
# Input: final_data/LARRY_data/intermediate_data_and_qc_details (output of build_larry_matrix.py). Output: <REBUILD>/FateMap/LARRY_data and .../LARRY_table1_whitelist; final_data is untouched.
# Usage: sbatch run_larry_table1.sbatch
set -euo pipefail
ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
FIG=$ROOT/final_figure_data/real_data_analysis_and_melanoma
RAW=${FIG5_RAW_DATA:-$FIG/raw_data}                                  # links to the raw data (see raw_data/)
FINAL=$FIG/final_data                                                # links to the final objects
REBUILD=${FIG5_REBUILD_DIR:-$FIG/analysis_data/preprocessing_rebuild}  # rebuilt files go here, never into final_data
CODE=$ROOT/code/TwINFER/paper_analysis/real_data_analysis_and_melanoma/preprocessing_code
PY=${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}   # conda activate does not work in batch jobs: call the environment python directly
LOGS=$REBUILD/logs; mkdir -p "$LOGS"
export TWINFER_PROJECT_ROOT=$ROOT
cd "$CODE/FateMap_FM01_FM06_Watermelon"
$PY rebuild_corrected_downstream.py larry
$PY rebuild_corrected_downstream.py larry_table1
cd "$CODE/LARRY"
SOURCE="$REBUILD/FateMap/LARRY_table1_whitelist/filtered" KNN=15 $PY larry_raw_annotate.py
$PY compare_larry_table1.py
