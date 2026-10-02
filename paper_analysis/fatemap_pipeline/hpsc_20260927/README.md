# hpsc_20260927

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| build_gene_exclusion_list.py (UNREVIEWED) | Build the PDF's "Gene filter (called pairs, before ranking; non-TF targets)" exclusion list for the hPSC_20260927 (endo_T0/endo_T1) TwinScor |
| build_gene_universe.py (UNREVIEWED) | Detected-gene universe for hPSC_20260927: genes detected (>0 counts) in >=5% of the 6,454 QC+singlet-filtered cells (endo_T0+endo_T1 combine |
| build_table1_pdf.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| clone_overlap_test.py (UNREVIEWED) | First test for the twinfer_diff (endo_T0/endo_T1) two-timepoint dataset: how many clones (singletCode-called) are common to T0 & T1 vs. priv |
| compute_qc_metrics.py (UNREVIEWED) | Per-cell QC metrics (total UMI counts, detected genes, mito%) for endo_T0 and endo_T1 Cell Ranger filtered_feature_bc_matrix.h5, computed vi |
| pca_clonal_distance.py (UNREVIEWED) | 50-PC Euclidean distance comparison: clonal (twin/sister, same clone_id) cell pairs vs non-clonal (random cross-clone) cell pairs, on hPSC_2 |
| plot_qc_dist.py (UNREVIEWED) | QC distribution plots (total UMI counts, detected genes, mito%) for endo_T0 vs endo_T1, from qc_metrics.csv (computed by compute_qc_metrics. |
| score_competitors_vs_chipatlas_perturbseq.py (UNREVIEWED) | Score the 4 competitor methods (rho, ppcor, pidc, grnboost2) against OUR OWN ground truth for hPSC_20260927 -- the ChIP-Atlas candidate pair |
| singlets_before_after_qc.py (UNREVIEWED) | singletCode singlet counts before vs after applying the RNA-QC threshold (n_genes >= 1000, pct_counts_mt <= 10) to the lineage-barcode sampl |
