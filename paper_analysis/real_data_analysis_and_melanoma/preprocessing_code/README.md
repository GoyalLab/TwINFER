# Preprocessing: from raw data to the matrices of Figure 5

These scripts turn the raw data of each dataset into the cell x gene count matrix and the clone (or sister-pair) labels that the Figure 5
analyses read (`../analysis_code/`). One folder per dataset; each folder's README lists the source of the raw data, the scripts in run order
and what they produce.

| Dataset | System | Raw data | Folder |
|---|---|---|---|
| FM01, FM06 | WM989 melanoma cells, FateMap barcodes (FM01: treated with vemurafenib; FM06: untreated) | GEO GSE233766 + FateMap lineage-barcode tables | `FateMap_FM01_FM06_Watermelon/` |
| Watermelon naive / lag / late | T47D breast cancer cells, Watermelon barcodes | Cell Ranger output + barcode table [accession to add] | `FateMap_FM01_FM06_Watermelon/` |
| hPSC T0 / T15 | human pluripotent stem cells, endoderm differentiation | sequencing run of this study [accession to add] | `hPSC_endo_yscher/` |
| LARRY day 2 / 4 / 6 | mouse LSK cells in vitro (Weinreb et al. 2020) | GEO GSE140802 (library zips + barcode FASTQ) | `LARRY/` |
| CellTag3 day 2.5 / 5.0 | mouse cells, CellTag-multi | GEO GSM6681127-GSM6681135 + the authors' clone tables | `CellTag3/` |
| Michaels T1 / T2 | human T-cell progenitors (Michaels et al. 2024) | GEO GSE248549 + TLB amplicon FASTQs (SRA SRR26941209-12, -15, -16, -19, -20) | `Michaels_T1T2/` |
| Novo Nordisk hPSC | hPSC endoderm, sister / same-well samples | the authors' Seurat object [reference to add] | `hPSC_Novo_Nordisk/` |
| DTT mouse embryo | E13.5 embryo, DNA Typewriter tree (Yu et al.) | GEO GSE341627 + github.com/shendurelab/dtt-mouse | `DTT_mouse/` |
| PeTrace | mouse embryos E7.5-E10.0 (E9.5 used) | Zenodo record 23088914 | `PeTrace/` |
| SpaceBar | melanoma xenograft sections, spatial barcodes | the authors' cell x gene tables [reference to add] | `SpaceBar/` |

## Setup
1. Python: the twinfer environment (`package/environment.yml`, then `pip install -e package/`; it includes scanpy, anndata and singletCode).
   Extra tools for some datasets only: Cell Ranger and bcl-convert (hPSC, only to start from the sequencer output), starcode (hPSC barcodes),
   R with Seurat and qs (Novo), python 3.10 with igraph < 0.10 (CellTag3 clone repair).
2. Put the raw data in one folder with the sub-folder names below (links are fine) and tell the scripts where it is: either
   `export FIG5_RAW_DATA=/path/to/raw_data`, or `export TWINFER_PROJECT_ROOT=/path/to/project` with the data in
   `<project>/final_figure_data/real_data_analysis_and_melanoma/raw_data`. All paths are defined in `fig5_paths.py`.

   | Sub-folder of raw_data | Content |
   |---|---|
   | `FateMap_GEO_and_barcode_tables` | GSE233766 matrices, `Processed_FM01_.../<FM>/step*StarcodeShavedReads.txt`, Goyal et al. cluster labels |
   | `Watermelon_cellranger_and_barcodes` | `T47D-<stage>-<rep>/` Cell Ranger output, `Watermelon_barcode_cellID_UMIcounts.csv` |
   | `hPSC_endo_cellranger_and_fastq_Hanxiao` | `fastq/`, `cellranger_out/` of the hPSC run |
   | `LARRY_authors_zips_and_barcode_fastq` + `LARRY_published_LSK_d2_d4_d6.h5ad` | the 15 library zips, the barcode FASTQ, the published annotated object |
   | `CellTag3_GEO_yscher_copy` | the 9 Cell Ranger h5 files, `all_samples.csv`, the authors' master clone table |
   | `Michaels_GEO`, `Michaels_TLB_fastq` | GEO matrices; TLB amplicon FASTQs |
   | `Novo_nordisk_hPSC_endo` | the Seurat `.qs` object |
   | `DTT_GEO_GSE341627`, `DTT_authors_repo_tree` | GEO matrices; the dtt-mouse repository (tree, support data) |
   | `PeTrace_zenodo_23088914`, `SpaceBar_authors_tables` | the authors' files |
3. Outputs go to `FIG5_REBUILD_DIR` (default `<figure>/analysis_data/preprocessing_rebuild/`), one sub-folder per dataset.

## Running
Each folder has a launcher (`run_*.sbatch` for SLURM; it can also be run with `bash`). Submit with an explicit log file, e.g.
`sbatch -o logs/%x_%j.out FateMap_FM01_FM06_Watermelon/run_rebuild_fullsheet.sbatch`. Order within a dataset is given in its README;
datasets are independent of each other (LARRY's stage-2 script lives in `FateMap_FM01_FM06_Watermelon/rebuild_corrected_downstream.py`).
`h5ad_identity.py` compares two `.h5ad` files element by element (used by the launchers to check a rebuild against a reference file).

## Known limitations
- Michaels clone labels: the clone-graph step depends on Python's hash seed, so a rerun gives about the same clones (1,359-1,372 vs 1,371) but not identical ids.
- Novo, PeTrace and SpaceBar start from the authors' processed objects (no raw reads are processed here).
- Cell Ranger needs the 10x GRCh38-2024-A reference (`CELLRANGER_REF`).
