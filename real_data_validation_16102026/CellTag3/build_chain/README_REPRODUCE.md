# CellTag3: scripts that build celltag_clones_repaired_nopad.h5ad (copied 2026-10-16 from yuval.scher's helpers/, unmodified)

Inputs (GEO GSE218329-style CellTag-multi data + authors' tables), as hard-coded in the scripts:
- /gpfs/projects/p32655/yscher/celltag_3_geo_data/  : 9 Cell Ranger filtered_feature_bc_matrix.h5 (GSM6681127 d2_5; GSM6681128-35 d5_1..8), all_samples.csv (CellTag UMI rows), hsc.rna&atac.r1&2_master_v2.csv (authors' master clone table)
- CellTag-multi-2023/clone_tables/ (authors' repository; used by build_celltag_schema_clones.py)
Run order (see run_rebuild_celltag.sbatch for the exact commands used in the rebuild; each script reads/writes datasets/celltag/ next to helpers/):
 1 build_nosinglet_dataset.py   -> celltag_nosinglet_counts.h5ad            (RAW COUNTS from the Cell Ranger matrices)
 2 finalize_nosinglet.py
 3 celltag_apply_qc.py          -> ..._nosinglet_counts_mito20_min200genes.h5ad
 4 build_corrected_clones.py    -> clone_assignments_corrected_*.csv  (Jaccard >= 0.6 on 2-24 tags, connected components)
 5 build_celltag_schema_clones.py -> celltag_clones_published_nopad.h5ad + _counts.h5ad
 6 celltag_clones_repaired.py   (authors' sparse-clone repair; needs igraph, adjustText, cassiopeia, shap, upsetplot)
 7 build_celltag_repaired_schema.py -> celltag_clones_repaired_nopad.h5ad (normalized X) + celltag_clones_repaired_nopad_counts.h5ad (integer counts, same 54,657 cells)
Validated 2026-10-16 (rebuild job 3952316): steps 1-5 reproduce yscher's intermediates exactly; steps 6-7 were NOT rerun (igraph etc. missing in yscher's interpreter).
Python used for 1-5: /hpc/software/mamba/24.3.0/bin/python3 (3.10). dataset_paths.py and qc.py are helper modules the chain imports.
