# fm06_powerseek

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| fm06_powerseek_step1_preprocess.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| fm06_powerseek_step1b_preprocess_h5ad.py (UNREVIEWED) | Re-derive the z-scored HVG data matrix from the canonical FM06_integrated.h5ad (lognorm layer = log1p(CP10k), already-flagged 2000-gene Seur |
| fm06_powerseek_step2_powerseek.py (UNREVIEWED) | Power-Seek gene ranking (blind to lineage) + efficient delta_g via Rayleigh-Ritz approximation, following Ghosh/Chakrabarti/Raju (2025) Meth |
| fm06_powerseek_step2_powerseek_h5ad.py (UNREVIEWED) | Power-Seek gene ranking (blind to lineage) + efficient delta_g via Rayleigh-Ritz approximation, following Ghosh/Chakrabarti/Raju (2025) Meth |
| fm06_powerseek_step3_spectral_cluster.py (UNREVIEWED) | Build a spectral embedding of cells from the cell covariance matrix restricted to the top Power-Seek-ranked "memory genes" (blind to lineage |
| fm06_powerseek_step3_spectral_cluster_h5ad.py (UNREVIEWED) | Build a spectral embedding of cells from the cell covariance matrix restricted to the top Power-Seek-ranked "memory genes" (blind to lineage |
| fm06_powerseek_step4_evaluate.py (UNREVIEWED) | Evaluate the blind spectral clustering / embedding against the held-out ground-truth lineage barcodes (fatemap_clone_singletcode). Lineage i |
| fm06_powerseek_step4_evaluate_h5ad.py (UNREVIEWED) | Evaluate the blind spectral clustering / embedding against the held-out ground-truth lineage barcodes (fatemap_clone_singletcode). Lineage i |
| fm06_powerseek_step5_recursive_bipartition.py (UNREVIEWED) | Push the memory-gene spectral clustering to finer resolution via recursive (divisive) spectral bipartition: at each node, re-center/re-scale |
| fm06_powerseek_step6_depth_sweep_eval.py (UNREVIEWED) | Evaluate the recursive bipartition tree against held-out ground-truth clones at every resolution (depth) from K=2 up to the finest leaves, t |
