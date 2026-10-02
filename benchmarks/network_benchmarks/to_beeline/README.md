# to_beeline

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| boolode_real_convert.py | Convert the BoolODE twin sims of B_cell_activation / EMT_real / Pluripotent_real (simulation_data/boolode_sims_replicates/<net>/replicate_<r |
| boolode_to_twinfer_format.py | Convert twin_similarity_sweep.py output (twin_final_states.csv) into the input format infer_with_twinfer() expects (TwINFER_function_scripts |
| build_t1_10_beeline_inputs.py | 2026-09-17: Generate BEELINE 'twin_paired' scheme inputs at t1=10,t2=20 (instead of t1=1,t2=20) for all 3 simulated benchmarks. The 'spread' |
| convert_network_sweep_to_beeline.py | [RECONSTRUCTED 2026-09-30] Convert TwINFER's network_sweep raw simulation trajectories into BEELINE-format inputs (ExpressionData.csv + Pseu |
| e13_pos100_to_beeline_format.py | Convert the 3 x 10 e13_pos100 simulations into BEELINE-format inputs, same layout/conventions as mixed_to_beeline_format.py: |
| matrix_to_edgelist.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| mixed_to_beeline_format.py | Convert finished mixed_network_sweep simulations into BEELINE-format inputs, mirroring the network_sweep_final_20260824 layout: |
| pluripotent_matrix_to_edgelist.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| regenerate_legacy_beeline_inputs.py | Regenerate BEELINE inputs (ExpressionData.csv + PseudoTime.csv + GroundTruthNetwork.csv) for GSD/HSC/mCAD/VSC, the 4 legacy TwINFER-Gillespi |
| regenerate_new_beeline_inputs_missing_reps.py | Extend BEELINE inputs for EMT/Pluripotent to cover the additional simulation replicates now available on disk beyond what BEELINE was origin |
| twinfer_to_boolode_format.py | Convert TwINFER raw simulation trajectories (for the datasets listed in DATASETS below) into BEELINE-format inputs (ExpressionData.csv + Pse |
| twinfer_to_boolode_format_multistate.py | Sibling of twinfer_to_boolode_format.py: converts the new multi-state real-network reruns (GSD/HSC/EMT IC-unset "multistate" + all-5 seeded- |
| verify_networksweep_conversion.py | Independently verify convert_network_sweep_to_beeline.py's output against the raw TwINFER simulation files and ground-truth topology matrice |
