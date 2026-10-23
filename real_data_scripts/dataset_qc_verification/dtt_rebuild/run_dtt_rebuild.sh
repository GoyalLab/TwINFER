#!/bin/bash
# Rerun DTT steps 01 (sister pairs), 04 (count extraction, 7 GSM files) and 05 (analysis matrix) from the ORIGINAL scripts, with only
# their output paths rewritten (sed) to a scratch tree, then diff against finalized_data/DTT_mouse_data.
set -euo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
SRC=$ROOT/analysis_data/lineage_tracing_mice_embryonic_organism/DTT/scripts
W=$ROOT/analysis_data/dataset_qc_verification/rebuild/DTT_mouse
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
mkdir -p $W/F/counts_wanted $W/D/results $W/code
# inputs not produced by steps 01/04/05 (copied, md5 recorded): column->gene table and the handout cell lists
cp $ROOT/finalized_data/DTT_mouse_data/matrix_column_to_ensembl.csv $W/F/
md5sum $ROOT/finalized_data/DTT_mouse_data/matrix_column_to_ensembl.csv $SRC/../inputs/wanted_cells.csv $SRC/../inputs/wanted_cells_with_cousins.csv > $W/inputs_not_rebuilt.md5
sed "s#OUT = '/home/gzu5140/TwINFER_KA/finalized_data/DTT_mouse_data'#OUT = '$W/F'#" $SRC/01_sister_pairs.py > $W/code/01.py
sed "s#OUT = R + 'finalized_data/DTT_mouse_data/counts_wanted/'#OUT = '$W/F/counts_wanted/'#" $SRC/04_extract_counts.py > $W/code/04.py
sed -e "s#F = R + 'finalized_data/DTT_mouse_data/'#F = '$W/F/'#" -e "s#e.to_csv(D + 'results/#e.to_csv('$W/D/results/#" $SRC/05_expressed_gene_list.py > $W/code/05.py
for f in 01 04 05; do grep -c "$W" $W/code/$f.py | sed "s/^/$f path rewrites: /"; ! grep -n "finalized_data" $W/code/$f.py | grep -v "^.*#" | grep -E "to_csv|save_npz|open\(.*'w'" ; done
cd $W/code
$PY 01.py > $W/01.log 2>&1
for g in $(ls $ROOT/real_data/DTT_mouse/GSE341627_RAW/*.df_cell.csv.gz | xargs -n1 basename | cut -d_ -f1); do $PY 04.py $g > $W/04_$g.log 2>&1 & done; wait
$PY 05.py > $W/05.log 2>&1
$PY $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification/dtt_rebuild/diff_dtt.py
