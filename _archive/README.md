
## 2026-09-30 (session 2): fatemap superseded scripts (user: yes)
- `paper_analysis/fatemap_pipeline/superseded_supplement_scripts/`: 13 `SUPERSEDED_*` scripts + the `SUPERSEDED_originals_unmodified/` copies (the pre-edit originals of the same files). Superseded by the `run_supp_gated_bootstrap_*` / `run_twinscore_*` family. No live script referenced them.
- `paper_analysis/fatemap_pipeline/build_hpsc_endoderm_qc_matrix_LEGACY_qcfirst.py`: legacy QC-first variant of the hPSC endoderm QC matrix builder.

## 2026-09-30 (session 2): filtered real-network / multistate inference drivers (user: keep no-filter)
`benchmarks/network_benchmarks/superseded_filtered_infer/`: `infer_network_simulation_real_network.py`, `infer_network_simulation_multistate.py`, `run_infer_multistate.sh`, `run_infer_network.sh` (alpha 0.01 / z_two_states 12 versions; outputs went to `twinfer_inference/`). Kept: the `*_nofilter.py` drivers and `run_infer_network_nofilter.sh` / `run_infer_all_nofilter.sh` (alpha 0.999999, outputs `twinfer_inference_nofilter/`). `to_beeline/regenerate_legacy_beeline_inputs.py` now imports the helpers from the no-filter module. Note: `analysis_plots/evaluate_real_networks_boolode.ipynb` text still says its TwINFER input came from the filtered run.

## 2026-09-30 (session 2): package launcher
`package/superseded/run_drift.sh` (from `package/twinfer/simulation/run_drift.sh`): stale duplicate of `simulations/drift_multiple_state/regenerate_drift_A_B.sh` (older job split); a package should not hold launchers. Archived (user: yes).

## 2026-10-01 (session 3): CellTag3 raw-data notebooks
`notebooks/real_data_analysis/`: `celltag_analysis.ipynb`, `normalization_comparison.ipynb` (user decision): they rebuild CellTag3 from raw GEO files (`real_data/celltag_3_geo_data/`) and write to the old `Keerthana_b1042/analysis_data/celltag_3` path. The only CellTag3 input now is `real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad` (already processed; no normalization or preprocessing). Archive keeps them as the record of how the h5ad was built.

## 2026-10-01 (session 3): ALL_PAIRS inference launchers
`benchmarks/network_benchmarks/infer/superseded_launchers/`: run_mixed_network_sweep_allpairs(.sh, _t1_10.sh), run_network_sweep_final_allpairs(.sh, _t1_10.sh), run_real_network_allpairs(.sh, _t1_10.sh), run_e13_pos100_allpairs_t1_10.sh. Same python invocation modulo array chunk / T1 variant; replaced by `run_infer_allpairs.sh <run>` + `runs/<run>.env` (smoke test tests/smoke/smoke_infer_allpairs_runs.py compares against these archived originals).

## 2026-10-01 (session 3): work_in_progress items (user decision: archive)
`simulations/pulsed_regulation/simulate_pulse.py` (pulsed-regulation simulation, no matching figure in the manuscript) -> `_archive/simulations/pulsed_regulation/`. `saturation_effects/` was already archived in `_archive/paper_analysis/parameter_scan/saturation_effects/`.
