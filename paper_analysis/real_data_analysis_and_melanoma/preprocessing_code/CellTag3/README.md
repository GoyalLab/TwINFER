# CellTag3 (mouse, CellTag-multi; day 2.5 and day 5.0): Cell Ranger matrices + CellTag reads -> count matrix with clone labels

**Raw data** (`raw_data/CellTag3_GEO_yscher_copy`): GEO GSM6681127 (day 2.5) and GSM6681128-GSM6681135 (day 5.0) Cell Ranger `filtered_feature_bc_matrix.h5`,
`all_samples.csv` (CellTag UMI table) and the authors' master clone table `hsc.rna&atac.r1&2_master_v2.csv`.

**Steps** (the scripts work in a folder that holds this folder as `helpers/` next to `datasets/` and `exports/`; the launchers set this up)
1. `build_nosinglet_dataset.py`, 2. `finalize_nosinglet.py`: cells and CellTags without singlet filtering.
3. `celltag_apply_qc.py`: mito <= 20 %, >= 200 genes (`qc.py` holds the rule).
4. `build_corrected_clones.py`: cells with 2-24 CellTags, Jaccard >= 0.6, connected components.
5. `build_celltag_schema_clones.py`: clone table onto the QC'd cells.
6. `celltag_clones_repaired.py` (+ the authors' helpers `sf_utils_morris.py`): the authors' sparse-clone repair (needs igraph < 0.10).
7. `build_celltag_repaired_schema.py` -> `celltag_clones_repaired_nopad.h5ad` (X = log1p CP10k, clone label `repaired_clone`) and `..._counts.h5ad`.
Launchers: `run_rebuild_celltag.sbatch` (1-7), `run_rebuild_celltag_steps67.sbatch` (6-7 only). `dataset_paths.py` resolves file names inside `datasets/`.

**Output**: 54,657 cells.
