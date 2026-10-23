# FateMap-style datasets (FM01/03/05/06/07/08, HS054, Watermelon): build scripts (copied 2026-10-16 from code/TwINFER, unmodified except where noted)

Inputs: Cell Ranger filtered matrices (GEO GSE233766 + the 'more data' tar) and the barcode tables under real_data/Fatemap_barcodes/ (RAW_DIR in fatemap_qc_utils.py).
Original builds (finalized_data/<ds>_data/qc_filtered): build_fm01_qc_matrix.py, build_fm06_qc_matrix.py, build_fm08_qc_matrix.py, build_fm_variants_qc_matrix.py (FM03|FM05|FM07), build_watermelon_qc_matrix.py, build_hs054_qc_matrix.py. Rebuilt byte-identical 2026-10-16 (rebuild_and_diff.py, run_rebuild_fm_family.sbatch).
Integrated h5ads: build_fm01/06/08_integrated.py, build_watermelon_stage_integrated.py, prep_watermelon_stage_qc.py.
singletCode criterion-4 rescue (as in the original builds, pools rows by cellID across samples): singletcode_cross_sample_rescue.py ; check_rescue_logic.py measures its effect.
Full-sheet / corrected rebuild (2026-10-16, requested by K. Arun): build_fm_fullsheet_rebuild.py (FM01/03/05/06/08: paper mito cutoffs 26/30/26/21/26, singletCode on the whole sheet in one call, rescue per (sample, cellID)); rebuild_corrected_downstream.py (Watermelon, LARRY, integrated h5ads); run_rebuild_fullsheet.sbatch, run_fs_all.sbatch.
Edit the paths at the top of fatemap_qc_utils.py / the build scripts (TWINFER_PROJECT_ROOT, RAW_DIR) to run elsewhere.
