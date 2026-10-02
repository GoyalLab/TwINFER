# network_analysis

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| check_multistate_at_t2000.py | Does the multistate / seeded population actually contain MULTIPLE STATES at the end of the before-division run (t=2000 for multistate, 800/1 |
| plot_real_network_final_clusters.py | Cluster cells by their *final-timepoint* expression state, then redraw the mean +/- std gene-expression trajectories (mRNA and protein) with |
| plot_real_network_trajectories.py | Plot mean gene-expression trajectories (mRNA and protein) across time for each real-network simulation, using the rep-0 `simulation_before_d |
| plot_trajectory_by_simtype.py | Compare per-gene mRNA / protein trajectories across the three simulation variants of the real-network sims, to see what actually changed bet |
| predict_multistability.py | Predict, WITHOUT running the Gillespie sims, how many distinct steady states each real-network TwINFER simulation should produce -- and cros |
| run_final_clusters.sh | 8 networks -> 8 worker processes (one CSV each, single-threaded per worker). |
