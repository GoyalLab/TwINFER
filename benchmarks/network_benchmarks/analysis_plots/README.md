# analysis_plots

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| benchmark_network_sweep.ipynb | Benchmark: Network Sweep -- TwINFER (with fan-out removal) vs. BEELINE methods |
| benchmark_sweep_heatmap_grid.py | One-figure benchmark grid rendered with the notebook's OWN plotting function. |
| benchmark_sweep_summary_plot.py | Combined network-sweep benchmark summary -- one figure, three metric panels. |
| benchmark_twinfer_vs_boolode.ipynb | TwINFER vs BEELINE benchmark - both simulation tracks |
| compare_sims.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| compare_twinfer_beeline.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| evaluate_networksweep_beeline.ipynb | Evaluate BEELINE network-sweep benchmark results |
| evaluate_networksweep_final_beeline.ipynb | Evaluate BEELINE network-sweep benchmark results |
| evaluate_real_networks_boolode.ipynb | Evaluate real-network benchmark results (full inference, all 8 networks) |
| full_comparison_table.py | 2026-09-17: Full unsigned+signed auprc_x comparison table -- TODO4v2(nogate), cross-corr-only, and all 7 BEELINE competitors -- across the 3 |
| grn_plot_spread_v2.py | ============================================================ |
| heatmap_data_mixed_sweep.py | Per-replicate data for mixed_sweep, same structure as heatmap_data_network_sweep.py: one column per topology instance (rep0/rep1/rep2), each |
| heatmap_data_network_sweep.py | Per-replicate data for the network_sweep heatmap (matching the reference PDF's layout, e17 dropped per user instruction): 4 topologies (grn_ |
| heatmap_data_real_networks.py | Per-network data for real_networks. NOTE: unlike network_sweep/mixed_sweep, real_networks has no '3 topology instances per family' structure |
| make_heatmap_pdf.py | Static PDF+PNG heatmap built DIRECTLY from summary_metrics_table_network_sweep_t1_1.csv (the canonical, already-reviewed network_sweep summa |
| make_heatmaps.py (UNREVIEWED) | Heatmaps (unsigned, directed only) matching heatmap_network_sweep_shared.pdf's style: methods as rows (sorted by avg rank, best on top), gro |
| motif_scoring_analysis.ipynb | Per-motif inference accuracy |
| network_sweep_analysis.ipynb |  |
| pearson_sign_table.py | 2026-09-17: Full table -- unsigned auprc_x, signed auprc_x, unsigned f1_topk, signed f1_topk -- for TODO4v2(nogate), cross-corr-only, and al |
| per_dataset_wins_t1_10.py | Per-dataset (per individual sim replicate) win/loss count: TODO4v2(nogate) auprc_x vs the BEST competitor's auprc_x on that SAME replicate ( |
| per_networktype_table_t1_10.py | Same format as the earlier 'unsigned auprc_x, t1=10' table (rows = method, values = mean auprc_x), but broken out per network/sim type (topo |
| per_replicate_heatmap_csvs_t1_1.py | One CSV per benchmark TYPE (network_sweep = e13_pos100 + network_sweep_final broader combined, mixed_sweep = mixed_network_sweep, real_netwo |
| plot_all_nofilter_heatmap.py | Heatmap comparison of the combined old+new, filtered+no-filter TwINFER-vs-BEELINE results (real_networks_all_nofilter_combined_scores.csv),  |
| plot_grn.py | Sample from 0.45 → avoids the near-white low end of each colormap |
| plot_mixed_network_sweep.py | Plot every ground-truth network in input_data/mixed_network_sweep/ using the same spread-layout style as generating_networks.ipynb (grn_plot |
| plot_n6_e9_rank_opacity.py | 3 GRN plots for grn_n6_e9_pos100_center_rep0 (rep0 instance, sim replicate 0, t1=1): ground truth, TODO4v2(nogate), and its best competitor  |
| plot_network_sweep_final_inferred.ipynb | Ground-truth networks for `network_sweep_final` |
| plot_seeded_trajectories.py (UNREVIEWED) | Mean +/- std gene-expression trajectories (mRNA and protein) for the SEEDED real-network sims, one line per steady-state basin the cells wer |
| plot_twin_vs_random.py (UNREVIEWED) | Twin-pair vs random-pair Euclidean distance, post-division, for BOTH the original ("initial") real-network sims and the seeded (balanced-bas |
| real_networks_summary_table.py | Metrics summary table for the 8 real networks (GSD, HSC, VSC, mCAD, B_cell_activation, Circadian_cycle, EMT, Pluripotent) -- TwINFER vs. the |
| summary_table_todo4v2.py | Reproduces the exact 'network_sweep_summary_metrics_table.csv' format the user provided (columns: variant, dataset_id, method, f1_topk, prec |
| test_grnboost_vsgenie_3.ipynb | GENIE3 vs GRNBoost2: isolating the `max_features` mechanism |
