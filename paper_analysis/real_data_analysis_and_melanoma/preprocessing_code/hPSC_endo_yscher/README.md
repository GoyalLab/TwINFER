# hPSC endoderm differentiation (T0, T15): sequencing run -> count matrix with clone labels

**Raw data** [accession to add] (`raw_data/hPSC_endo_cellranger_and_fastq_Hanxiao`): BCL / FASTQ of the run, Cell Ranger output of samples endo_T0 and endo_T1 (= T15),
barcode FASTQs; stagger sheets in this folder (`staggers_B_sheet_index.txt`, `TwINFER_Diff_staggers.txt`).

**Steps**
1. `bclconvert_20260924_TwINFER_Diff.sh` (sample sheet `20260924_TwINFER_Diff_bclconvert.csv`): BCL -> FASTQ.
2. `run_cellranger.sh`: Cell Ranger count per sample (needs `CELLRANGER_REF`, 10x GRCh38-2024-A).
3. Barcode pipeline: `barcode_pipeline_twinfer_diff/stepOne.py`, `stepTwo.R`, `barcode_pipeline_TwINFER_2026/stepThree.py` (starcode, distance 8) -> per-sample barcode tables.
4. `build_endo_dataset.py`: cells with mito < 20 % and >= 200 genes; clone label `clone_c5` = singletCode (UMI cutoff 5) on the barcode tables ->
   `endo_counts_mito20_min200genes.h5ad` (Cell Ranger counts; the second output, a kb-python spliced/unspliced file, is not used in Figure 5).
Launcher for 3-4 with a comparison to the reference files: `run_rebuild_hpsc_endo.sbatch`.

**Output**: T0 21,244 cells, T15 5,615 cells; 1,246 T15 cells carry a clone label (109 clones with >= 2 cells).
