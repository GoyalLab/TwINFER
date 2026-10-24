"""Where the data of Figure 5 (real_data_analysis_and_melanoma) lives: the one place the preprocessing scripts take their paths from.

Figure   : real_data_analysis_and_melanoma (Figure 5), preprocessing_code.
Purpose  : every build script imports these constants instead of writing absolute paths. Raw inputs are read through the symbolic links in
           raw_data/ (also the files that belong to other people's folders, e.g. yscher's and Hanxiao's); rebuilt outputs are written to
           REBUILD, never to final_data, so that a rerun can be compared with the final files without overwriting them.
Layout   : <PROJECT_ROOT>/final_figure_data/real_data_analysis_and_melanoma/
             raw_data/      links to the raw downloads (+ external_tools/: starcode, the fatemapbc environment, the python 3.10 of the shared mamba)
             final_data/    links to the final objects
             input_folder/  small input tables
             analysis_data/preprocessing_rebuild/   output of the build scripts when they are rerun (REBUILD)
Environment variables (all optional)
           TWINFER_PROJECT_ROOT  project root (default /gpfs/projects/b1255/hzhang/TwINFER_KA)
           FIG5_RAW_DATA         raw_data folder (default <figure>/raw_data)
           FIG5_REBUILD_DIR      output folder of rebuilds (default <figure>/analysis_data/preprocessing_rebuild)
Usage    : scripts of the sub-folders do  sys.path.insert(0, <preprocessing_code>)  and  from fig5_paths import RAW, FINAL, REBUILD, ...
"""
import os

PROJECT_ROOT = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")
FIGURE = os.path.join(PROJECT_ROOT, "final_figure_data", "real_data_analysis_and_melanoma")
RAW = os.environ.get("FIG5_RAW_DATA", os.path.join(FIGURE, "raw_data"))
FINAL = os.path.join(FIGURE, "final_data")
INPUT = os.path.join(FIGURE, "input_folder")
REBUILD = os.environ.get("FIG5_REBUILD_DIR", os.path.join(FIGURE, "analysis_data", "preprocessing_rebuild"))

# raw data (links, see raw_data/)
FATEMAP_RAW = os.path.join(RAW, "FateMap_GEO_and_barcode_tables")             # GEO GSE233766 matrices, starcode barcode tables, Goyal et al. cluster labels
WATERMELON_RAW = os.path.join(RAW, "Watermelon_cellranger_and_barcodes")
LARRY_RAW = os.path.join(RAW, "LARRY_authors_zips_and_barcode_fastq")
CELLTAG_GEO = os.path.join(RAW, "CellTag3_GEO_yscher_copy")
HPSC_HANXIAO = os.path.join(RAW, "hPSC_endo_cellranger_and_fastq_Hanxiao")
HPSC_SCRIPTS_2026 = os.path.join(RAW, "Hanxiao_TwINFER_2026scRNAseq")
# data of yscher's project tree that the build scripts read (one link per folder; the scripts themselves are copied into this repository)
MICHAELS_GEO = os.path.join(RAW, "Michaels_GEO")                                 # GEO matrices of T1 and T2 (+ pools); byte-identical to the copy in yscher's michaels2024/raw
MICHAELS_TLB_FASTQ = os.path.join(RAW, "Michaels_TLB_fastq")                     # TLB amplicon FASTQs (SRA), only available through yscher's folder
MICHAELS_CLONES = os.path.join(FINAL, "Michaels_T1T2_data", "cell_to_clone.csv")  # yscher's clone labels (md5-identical copy in final_data)
MICHAELS_QC_H5AD = os.path.join(FINAL, "Michaels_T1T2_counts_yscher")            # yscher's QC h5ads michaels_T1_qc.h5ad / michaels_T2_qc.h5ad (final_data link)
YSCHER_ENDO = os.path.join(RAW, "yscher_twinfer_diff_endo")                      # stepThree tables, index_fix/B outputs, kb_nac/ of the hPSC endoderm build
YSCHER_CELLTAG = os.path.join(RAW, "yscher_celltag_datasets")                    # her CellTag intermediates / final files (reference for the rebuild)
YSCHER_CELLTAG_EXPORTS = os.path.join(RAW, "yscher_celltag_clone_exports")       # her clone tables (reference for the rebuild)
NOVO_RAW = os.path.join(RAW, "Novo_nordisk_hPSC_endo")                           # the authors' Seurat object (.qs)
DTT_GEO = os.path.join(RAW, "DTT_GEO_GSE341627")
DTT_REPO = os.path.join(RAW, "DTT_authors_repo_tree")
PETRACE_RAW = os.path.join(RAW, "PeTrace_zenodo_23088914")
TOOLS = os.path.join(RAW, "external_tools")
STARCODE = os.path.join(TOOLS, "starcode")
FATEMAPBC_ENV = os.path.join(TOOLS, "fatemapbc_env")
MAMBA_PYTHON = os.path.join(TOOLS, "mamba_python3.10")                          # python 3.10 of the shared mamba, used for yscher's scripts
SEURAT_ENV = os.path.join(TOOLS, "seurat_qs_env")                                # conda environment with R, Seurat and qs (Novo export)
