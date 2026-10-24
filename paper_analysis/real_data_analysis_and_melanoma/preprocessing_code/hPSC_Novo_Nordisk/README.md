# Novo Nordisk hPSC endoderm lineage tracing: the authors' Seurat object -> per-sample count matrices with clone labels

**Raw data** [reference to add] (`raw_data/Novo_nordisk_hPSC_endo`): `seurat_lineage_tracing_linneberg_agerholm_2026.qs`.

**Steps** (launcher `run_rebuild_novo.sbatch`)
1. `export_seurat.R` (R with Seurat and qs): exports counts, metadata, reductions and graphs of the object to neutral files.
2. `build_qc_per_sample.py`: one folder per sample (hESCs_p2_parent, hESCs_p3_same, hESCs_p3_sis, PS_sis, DE_same, DE_sis) with `counts.npz`
   (raw counts, cells x genes), `genes.txt`, `obs_metadata.csv`; only cells with a clone id, >= 1,000 genes and <= 10 % mitochondrial reads;
   clone label `fatemap_clone_singletcode` = the authors' `clone.id`, cell types `clust_annot_02`.

Used in Figure 5 for the covariance decomposition only.
