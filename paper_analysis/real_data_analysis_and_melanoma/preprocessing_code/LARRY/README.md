# LARRY (mouse LSK cells in vitro, days 2, 4, 6; Weinreb et al. 2020): raw data -> count matrix with clone labels and cell types

**Raw data**: GEO GSE140802 - the 15 inDrops library zips `LSK_d2_*`, `LSK_d4_*`, `LSK_d6_*` and `LARRY_sorted_and_filtered_barcodes.fastq.gz`
(`raw_data/LARRY_authors_zips_and_barcode_fastq`), and the authors' annotated object `LSK_d2_d4_d6.h5ad` (`raw_data/LARRY_published_LSK_d2_d4_d6.h5ad`).

**Steps**
1. `build_larry_matrix.py` (launcher `run_build_larry_matrix.sh`): combines the 15 libraries into one matrix; lineage barcodes per (library, cell, UMI)
   with >= 10 reads, merged within Hamming distance 3 -> `<REBUILD>/LARRY/processed/` (matrix + `umi_table.csv`).
2. `../FateMap_FM01_FM06_Watermelon/rebuild_corrected_downstream.py larry_table1` (runs `build_final_matrix_umi3.py` of this folder): cell whitelist with the
   minimum UMIs of Weinreb et al. Table 1 (day 2 >= 500, days 4/6 >= 300), singletCode (min 3 UMIs) + cross-sample rescue, mito <= 20 %, >= 200 genes.
   95,584 -> 33,362 cells, 5,499 clones. (`larry` instead of `larry_table1` uses a dip-test + Otsu whitelist; `compare_larry_table1.py` compares the two.)
3. `larry_raw_annotate.py`: cell types from the published object where the cell is in it, k-NN transfer (k = 15) otherwise; `annotation_source` says which.
Launcher for 2-3: `run_larry_table1.sbatch`.

**Output**: `filtered/` (raw counts, clone label `larry_clone_singletcode`) and `annotated/filtered_annotated_k15.h5ad`.
