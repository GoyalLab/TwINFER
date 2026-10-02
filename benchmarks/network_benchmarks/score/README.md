# score

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| Twinfer_nogamma_ablation.py | Ablation: drop new_gamma from todo4v2_score's nogate variant, per the finding that new_gamma (z_gamma) and s_zdagger (z_dagger) are correlat |
| Twinfer_sim_scoring.py | Apply TODO4v2 (both variants: old=signed-REG/one-sided-gate, new=abs-REG/two-sided-gate -- see paper_analysis/larry_hematopoiesis_validation |
| Twinfer_sim_scoring_t1_10.py | t1=10,t2=20 variant of todo4v2_sim_scoring.py (Part 6 of handoff/2026-09-18_todo4v2_signed_metrics_and_t1_10_pipeline.md) -- glob-path chang |
| apply_twinscore_supplement.py | First test of the 2026-09-17 TwinScore supplement's continuous formula (handoff/... TwinScore_supplement.pdf) on the e13_pos100 network_swee |
| beeline_scoring_extract.py | RECONSTRUCTED 2026-09-30 (AST extraction from benchmark_network_sweep.ipynb; cell number above each definition). The original /tmp/beeline_s |
| boolode_real_todo4v2_score.py | todo4v2 (old/new/nogate) scoring of the BoolODE real-network all-pairs inference JSONs, plus a side-by-side with the TwinScore_supplement (g |
| motif_scoring.py | Per-motif GRN-inference accuracy scoring. |
| partial_corr_term.py | Conditional (partial-correlation) term for current_score, added 2026-09-18 to address the VSC indirect-correlation confound (see handoff/202 |
| scaling_analysis.py | 2026-09-17: What scales with network size / density / negative-edge-fraction, pooled across ALL replicates of ALL three simulated benchmarks |
| scaling_analysis_t1_10.py | t1=10 variant of scaling_analysis.py -- glob-path change only, reusing every function (net_properties, cohens_d, process_one) unchanged. Par |
| score_all_nofilter_combined.py | Combined AUPRC/F1 scoring across all 16 real-network datasets (the original 8 pre-multistate-work networks + the 8 multistate/seeded dataset |
| score_autoreg.py | Scores autoregulation's 40 TwINFER JSONs and 40x7 BEELINE rankedEdges.csv against ground truth, reusing the same extracted notebook scoring  |
| score_beeline_20260824.py | Scores the 150x7 BEELINE rankedEdges.csv results against ground truth, producing beeline_analysis_output.csv for the 20260824 rerun. |
| score_beeline_real_networks_t1_10.py | Competitor (BEELINE) scoring for real_networks at t1=10 -- previously unscored because score_mixed_network_sweep.py's generic score_beeline( |
| score_beeline_t1_10.py | Regenerate beeline_analysis_output.csv-equivalent files for the t1=10 rerun (Part 6 of handoff/2026-09-18_todo4v2_signed_metrics_and_t1_10_p |
| score_cyclic_g3456.py | Scores TwINFER and BEELINE results for the cyclic_g3/g4/g5 benchmark (simple directed cycles, 3/4/5 genes). Reuses benchmark_network_sweep.i |
| score_e13_pos100.py | Scores TwINFER and BEELINE results for the e13_pos100 benchmark (3 of the network_sweep_final OFAT "center" topologies, n6_e13_pos100_rep{1, |
| score_mixed_network_sweep.py | Scores TwINFER and BEELINE results for the mixed_network_sweep benchmark (cycle-core + linear + sprinkled-autoregulation networks, n=6/n=10  |
| score_multistate_benchmark.py | Score the multistate/seeded real-network benchmark: 7 BEELINE algorithms (PIDC, GENIE3, GRNBOOST2, PPCOR, SCODE, SCSGL, PEARSON) from beelin |
| score_original_benchmark_twinscore.py | Rate TwINFER on the ORIGINAL 8 real-network simulations (GSD/HSC/VSC/mCAD/ EMT/Pluripotent/B_cell_activation/Circadian_cycle -- the pre-mult |
| score_real_networks_analytic_tuned.py | Score the 8 original real-network simulations with the ANALYTIC, yscher-tuned TwinScore (the one built for the LARRY panels in paper_analysi |
| score_real_networks_analytic_zhet_fixed.py | Rebuild real_networks_all_nofilter_heatmap.pdf with the TwINFER row computed from ANALYTIC z-scores instead of the 10k-shuffle permutation z |
| score_real_networks_summary.py | Metrics summary tables for the real-network benchmark (GSD, HSC, VSC, mCAD, B_cell_activation, Circadian_cycle, EMT, Pluripotent) -- BEELINE |
| score_twinfer_20260824.py | Scores the 150 TwINFER JSON results (rerun_twinfer_150.py's output) against ground truth, producing twinfer_analysis_output.csv for the 2026 |
| signed_scoring_analysis.py | 2026-09-17: Signed edge-detection metrics (existence AND correct activation/repression sign) for the 3 simulated-GRN benchmarks, per user in |
| signed_scoring_analysis_t1_10.py | t1=10 variant of signed_scoring_analysis.py -- glob-path change only, reuses run()/ score_one_signed()/competitor_signed_summary() unchanged |
| test_alt_statistics.py (UNREVIEWED) | Compare 4 alternative Step-1 test statistics against the current Spearman baseline, on the SAME cell partition the pipeline actually uses fo |
| todo4v2_sim_scoring.py | TODO4v2 scoring for the simulated benchmarks (network_sweep_final, e13_pos100, mixed_network_sweep, real_networks) against their known groun |
| twinfer_scoring_extract.py | RECONSTRUCTED 2026-09-30 (AST extraction from benchmark_network_sweep.ipynb; cell number above each definition). The original /tmp/twinfer_s |
| zdagger_signal_strength_larry_vs_sims.py | Part 5 follow-up (handoff/2026-09-18_todo4v2_signed_metrics_and_t1_10_pipeline.md, open item 3): tests the "real-data-noise vs clean-sim" hy |
| zscore_usefulness.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| zscore_usefulness_auprc.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
