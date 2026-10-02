# simulate

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| autoregulation.py | Set output path |
| check_twinfer_sim.ipynb |  |
| cycle.py | path_to_output_folder = f'{TWINFER_PROJECT_ROOT}/simulation_data/cycle_data/'   [2026-09-30 replaced per user: the original name no longer e |
| cyclic_g3_g4_g5_sim.py | Set output path |
| mixed_network_sweep_sim.py | Gillespie simulations for every network in input_data/mixed_network_sweep/, set up the same way as the network_sweep_final sims (same base c |
| network_sweep.py | ========================================================== |
| oscillation_simulation.py | Set output path |
| pluripotent_sim_6000steps.py | Vanilla (default-parameter, non-multistate) rerun of the Pluripotent real-network sim at 6000 before-division steps, instead of the original |
| real_network.py | ========================================================== |
| real_network_multistate_sim.py | Re-run the TwINFER Gillespie sims for the real networks that CAN be pushed into a multi-stable regime, using per-network (Hill n, k_add) tun |
| real_network_seeded_sim.py | Seeded-initial-condition variant of real_network_multistate_sim.py. |
| real_network_sim.py | Set output path |
| regulated_mutual_negative_simulate.py | Set output path |
| run_figure_3_1k_simulation.py | Standalone driver for generating the 4 figure_3 networks not already covered by figure_2 (A_rep_B, A_and_B_both_repress, A_rep_B_B_to_A, A_a |
| run_figure_3_1k_slurm_array.sh | Generates the 4 figure_3 networks not already in figure_2 (A_rep_B, |
| run_mixed_network_sweep_sim.sh | 48 networks x 3 replicates = 144 simulations, split round-robin across 6 |
| run_multistate_sims.sh | IC-unset (empty start) sims for the real networks that CAN be pushed multistate, |
| run_pluripotent_6000steps.sh | Vanilla (non-multistate) Pluripotent rerun at 6000 before-division steps |
| run_row6_vs_row4_smoketest.sh | Diagnostic only (see test_row6_vs_row4_smoketest.py docstring): 2 reps each |
| run_seeded_sims.sh | Seeded-IC sims: 6000 cells split evenly across each network's k stable basins, |
| run_simulations_for_figures.sh | ---- Conda activation that works both on Quest and inside Singularity |
| synthetic_network.py | ========================================================== |
| synthetic_network_high_density.py | ========================================================== |
| test_row6_vs_row4_smoketest.py | One-off diagnostic: does the CURRENT simulation code, run with the CURRENT median_parameter.csv row 6 (k_add=2), reproduce the OLD figure_3_ |
