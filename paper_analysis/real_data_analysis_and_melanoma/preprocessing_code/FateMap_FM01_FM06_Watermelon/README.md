# FM01, FM06 (WM989 melanoma) and Watermelon (T47D): raw data -> count matrices with clone labels

**Raw data** (`raw_data/FateMap_GEO_and_barcode_tables`, `raw_data/Watermelon_cellranger_and_barcodes`)
- FM01: GEO GSE233766, GSM7434407 / GSM7434408 (replicates A, B; 1 uM vemurafenib). FM06: GSM7434419 / GSM7434420 (untreated).
- Lineage-barcode tables of the FateMap study: `Processed_FM01_FM03_FM04_FM05_FM06_FM08_FM09_FM10/<FM>/stepFour|stepThreeStarcodeShavedReads.txt` (column `BC50StarcodeD8`).
- Watermelon: Cell Ranger output of six samples (T47D naive, lag, late x 2 replicates) and `Watermelon_barcode_cellID_UMIcounts.csv`.

**Steps** (run from this folder)
1. `build_fm_fullsheet_rebuild.py FM01|FM06 <out>` -> `<out>/<FM>_data/qc_filtered/` (raw counts `.mtx`, genes, cells, `obs_metadata.csv` with the clone label `fatemap_clone_singletcode`).
   singletCode is run once on the whole barcode sheet (all samples), multiplets whose barcode combination recurs in another sample are rescued
   (`singletcode_cross_sample_rescue.py`), then cells are filtered on mitochondrial content (FM01 26 %, FM06 21 %) and gene counts (200-7,200; FM06 200-5,000).
   Launcher: `run_rebuild_fullsheet.sbatch`.
2. `rebuild_corrected_downstream.py wm_qc`, then `wm_split`, then `wm_stage naive|lag|late`: the same QC for Watermelon (mito 20 % naive, 25 % lag/late),
   split into the three stages, one integrated `.h5ad` per stage.
3. `rebuild_corrected_downstream.py fm_integrated FM01|FM06`: integrated `.h5ad` (log1p CP10k, PCA, Harmony/Scanorama embeddings; counts kept in `layers['counts']`).
   Launcher for 2 and 3: `run_fs_all.sbatch` (after step 1).
Helpers: `fatemap_qc_utils.py` (loading, QC filters, sample sheet), `fatemap_integration_utils.py`; `build_fm01_qc_matrix.py`, `build_fm06_qc_matrix.py`,
`build_watermelon_qc_matrix.py` hold the per-dataset settings; `check_rescue_logic.py` reports what the rescue changes.

**Output** (`<FIG5_REBUILD_DIR>/FateMap/`): FM01 10,475 cells / 4,583 clones; FM06 12,865 / 5,394; Watermelon naive 10,322, lag 12,689, late 10,268 cells.
