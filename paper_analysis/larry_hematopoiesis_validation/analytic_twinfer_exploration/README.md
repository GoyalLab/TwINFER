# analytic_twinfer_exploration

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| analytic_twinfer.py (UNREVIEWED) | Shuffle-free ("analytic") TwINFER for the new LARRY gene sets. |
| analytic_twinfer_unw.py (UNREVIEWED) | Shuffle-free ("analytic") TwINFER for the new LARRY gene sets. |
| analytic_variants.py (UNREVIEWED) | Shuffle-free TwINFER, two knobs: correlation weighting (clone vs unweighted) and TwinScore composition (default vs yscher's tuned `twinscore |
| analytic_z_check.py (UNREVIEWED) | Test the analytic z approximation z ~= rho_obs * sqrt(m_eff - 1) (null mean 0, null SD = 1/sqrt(m_eff-1)) against the 5000-shuffle z-scores  |
| analytic_z_full.py (UNREVIEWED) | Analytic approximation of EVERY z-score infer_with_twinfer returns, vs the 5000-shuffle permutation z, for the NEW LARRY gene sets. Runs on  |
| analytic_z_larry.py (UNREVIEWED) | Analytic z vs the saved permutation null for the LARRY superseded (old) gene sets. For each pair/step: reconstruct z_emp = (rho_obs - mean(n |
| bench_core.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| bench_nb.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| compare.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| edge_type_bias.py (UNREVIEWED) | Which true edges does TwINFER (z_het-fixed twinScore) recover, by edge type? Pool GSD/HSC/VSC/mCAD/EMT. For each true edge: percentile rank  |
| edit_benchmark_nb.py (UNREVIEWED) | Adapt benchmark_methods.ipynb: add analytic_twinfer + perm_twinfer as methods, read the current _allpairs (not _50core) infer dirs, skip any |
| edit_nb.py (UNREVIEWED) | Rewrite pick_gene_sets.ipynb to follow helpers/build_panel_set.py's edge-selection method, keeping the 15-TF / 4-target recipe and dropping  |
| edit_nb_rankby.py (UNREVIEWED) | correlation criterion: rank the 15 TFs by MEDIAN /rho/ of their in-band edges, not edge count. |
| hybrid_twinfer.py (UNREVIEWED) | Hybrid TwINFER: analytic z's for everything EXCEPT z_het / z_d_het, which come from a cheap 200-shuffle permutation (the two terms twinScore |
| nb2.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| nb3.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| nb_as_script.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| revert_hvg.py (UNREVIEWED) | Revert the variability criterion to the original design: top-N_HVG genes split into equal-count dispersion thirds. Keep the current edges_to |
| run6.sh (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| signed_rule_test.py (UNREVIEWED) | Signed z_het-fixed twinScore with a NEW edge-sign rule: |
| standardization_test.py (UNREVIEWED) | Does the per-term panel standardization s() help? z_het-fixed twinScore computed 4 ways, scored AUPRC/random on the real-net sims (orig 7 +  |
| tuned_nogate_realnets.py (UNREVIEWED) | Tuned analytic twinScore WITHOUT the z_reg_gated gate -- the raw composition TS = E1 - s/drho/ - s/rho_d1/ - s/drr/ - s(min/rho_x/) - s/gamm |
| tuned_plus_zhetfix.py (UNREVIEWED) | z_het-fix ON TOP OF the yscher-tuned analytic twinScore, for (a) our 9 LARRY gene sets -> TFs x panel (b) yscher's 9 exact panels -> genes x |
| twinscore_matrix_realnets.py (UNREVIEWED) | twinScore matrix (gene_1 = source rows x gene_2 = target cols), mean over the no-filter replicates, for GSD / VSC / mCAD / HSC. True regulat |
| twinscore_matrix_realnets_tuned.py (UNREVIEWED) | Same matrix plot but with the TUNED analytic twinScore (score_real_networks_analytic_tuned.analytic_tuned_scores): E1 - CHG - RD1 - DR - XS  |
| yscher_panels_tuned.py (UNREVIEWED) | Analytic TwINFER (unweighted + tuned + gate) on yscher's EXACT 9 panels, cp10k input. Score by yscher's rule (genes x genes, all CollecTRI e |
| yscher_zhet_fixed.py (UNREVIEWED) | z_het-fixed analytic twinScore on yscher's EXACT 9 panels. |
| zdd_weight_sweep.py (UNREVIEWED) | Sweep the coefficient on the signed-cross-corr-difference term +C * s(rho_cross_xy - rho_cross_yx) in the yscher-tuned analytic twinScore (d |
| zscore_box_per_network.py (UNREVIEWED) | Per network: all 7 TwINFER z-scores on the x-axis, paired box+dots for true directed edges (green) vs non-edges (grey). Permutation z from t |
| zscore_true_false.py (UNREVIEWED) | Each TwINFER z-score, true vs. false directed edges, per network. Permutation z-scores straight from the no-filter ranked_edges (all n(n-1)  |
