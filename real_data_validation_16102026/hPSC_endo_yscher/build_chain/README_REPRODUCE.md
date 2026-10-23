# hPSC_endo_yscher: scripts copied 2026-10-16 (unmodified) from yscher's repo and Hanxiao's folders

Chain (counts used in the analysis = Cell Ranger counts, endo_counts_mito20_min200genes.h5ad):
 0 bclconvert_20260924_TwINFER_Diff.sh (+ 20260924_TwINFER_Diff_bclconvert.csv)   BCL -> FASTQ                          (not rerun)
 1 run_cellranger.sh                                                                RNA FASTQ -> Cell Ranger outs/          (not rerun; outputs used as given)
 2 barcode_pipeline_twinfer_diff/stepOne.py, stepTwo.R, barcode_pipeline_TwINFER_2026/stepThree.py (starcode D8)
   + TwINFER_Diff_staggers.txt / staggers_B_sheet_index.txt                         barcode FASTQs -> stepThree tables        (rerun, byte-identical)
 3 build_endo_dataset.py                                                            Cell Ranger h5 (mito<20%, >=200 genes) + stepThree (singletCode cutoff 5 -> clone_c5)
                                                                                    -> endo_counts_mito20_min200genes.h5ad (Cell Ranger counts; rerun, identical)
                                                                                    and endo_kbnac_spliced_unspliced.h5ad (kb layers; not used for the analysis)
 run_rebuild_hpsc_endo.sbatch = the rerun commands (inputs, interpreters and paths used 2026-10-16)
Paths inside the scripts point to /projects/b1042/GoyalLab/Hanxiao/... and /gpfs/projects/b1255/yscher/Transcriptomic Distance/; edit them to run elsewhere. The starcode binary is at /projects/b1042/GoyalLab/Hanxiao/twinfer_diff/scripts/starcode/starcode.
The kb count --workflow=nac step is called from build_endo_dataset.py's inputs (KB path); the kb count command itself was not found in the folders searched.
