#!/bin/bash
# Figure 5 preprocessing, DTT mouse: rerun of step 01 (sister pairs from the dated tree), 04 (counts of the analysed cells, one task per GEO matrix) and 05 (analysis matrix),
# then element-wise comparison with final_data/DTT_mouse_data (diff_dtt.py). Output: <REBUILD>/DTT_mouse/F (final_data is untouched).
# Usage: bash run_dtt_rebuild.sh   (or: sbatch run_rebuild_dtt.sbatch)
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
W=$REBUILD/DTT_mouse; export DTT_OUT=$W/F
mkdir -p "$DTT_OUT/counts_wanted"
cd "$CODE/DTT_mouse"
md5sum "$FIG/input_folder/wanted_cells.csv" "$FIG/input_folder/wanted_cells_with_cousins.csv" "$FINAL/DTT_mouse_data/matrix_column_to_ensembl.csv" > "$W/inputs_not_rebuilt.md5"   # inputs that steps 01/04/05 do not produce
$PY 01_sister_pairs.py > "$W/01.log" 2>&1
for g in $(ls "$RAW"/DTT_GEO_GSE341627/*.df_cell.csv.gz | xargs -n1 basename | cut -d_ -f1); do $PY 04_extract_counts.py $g > "$W/04_$g.log" 2>&1 & done; wait
$PY 05_expressed_gene_list.py > "$W/05.log" 2>&1
$PY diff_dtt.py
