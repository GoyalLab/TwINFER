# real_networks

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| fanout_mutual_z_boxplot.py | Plot Figure 4 fan-out and gated-regulation Z-scores for all gene pairs. |
| full_formula_v2_real_data.py | Real-data heatmap data: existdir and full (flat, permutation z_fanout) AUPRC + F1(top-k) for all 7 real_world_networks, same 'full' formula  |
| full_gated_fanout.py | full_gated = existdir + (PENALTY if z_fanout > THR else 0) + 0.5*s(z_d_het) -- replaces full's flat 1.0*s(z_fanout) term with the gated vers |
| full_gated_fanout_new_networks.py | full_gated (fixed penalty=-2.0) on the three newly-discovered real_data networks (EMT/B_cell_activation/Pluripotent), using their paper_anal |
| plot_zfanout_fanout_only.py | z_fanout_cross for the Fan_out motif only, t1=1h vs t1=10h, showing the confound-signature decay: at t1=1h the confounded non-edge (2,3/1) s |
| plot_zfanout_figure4.py | Box+dots of z_fanout(x,y) per undirected gene pair, for each figure_4 3-gene motif (Fan_out / Feed_forward / Mutual_regulation), at t1=1h an |
| zfanout_cross_figure4.py | z_fanout using CROSS-CORRELATION (not co-expression) -- the "(or cross corr)" alternative from the original definition. z_c(C,x) = max(/rho_ |
| zfanout_e13_pos100.py | sys.path.insert(0, '.') |
| zfanout_embedded_permutation.py | Does the GENUINE permutation-based z_fanout (the actual reference-script machinery, identify_actual_directed_edges with real shuffles) corre |
| zfanout_embedded_precision.py | Precision check for the genuine permutation-based z_fanout on embedded networks: for EVERY directed pair (not just the Fan-out-motif triads) |
| zfanout_figure4.py | Does z_fanout(x,y) = max_C min(z_rho(C,x), z_rho(C,y)) help discriminate true directed edges from non-edges on the figure_4 3-gene motif swe |
| zfanout_gated_test.py | Gated z_fanout: score(x,y) = existdir(x,y) + PENALTY if z_fanout(x,y) > THR, else existdir(x,y) UNCHANGED. Tests whether only intervening wh |
| zfanout_sign_check_figure4.py | Does z_fanout need to be ADDED or DELETED (subtracted/excluded) on top of the existdir (existence+direction) formula, on the cleanest ground |
