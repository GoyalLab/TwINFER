# TwinScore stage checks (Figure-2 and drift simulations)
Per-replicate TwinScore z-scores (PDF Sec. 4,7,8,11) and variants of the unshared test z_U.
Design: sample B = 20 h (1,500 twin clones + 3,000 singletons), sample A = 1 h, 3,000 shared clones for direction.
- `figure2_1000_zscores.py`  z-scores for 4 Figure-2 scenarios (single/two-state x A_B/A_to_B), 100 replicates
- `drift_zscores.py`         same on `simulation_data/drift_lowhigh_Kramp` (k_on ramped 0-10 h to 0.12/1.66)
- `u_variants.py`            z_U variants: S from twins vs all cells; ranks vs log1p; bootstrap sd
- `u_time_pool.py`           z_U vs sampling time and Stouffer pooling over disjoint twin clones
- `plot_zscores.py`, `plot_zscores_with_drift.py`  box plots + rates
- `interactive_scripts/`     exploratory scripts from the session (calibration, classification, two-sample design)
Run with the twinfer-code python. Outputs: `analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores/`
(see `_paths.py`). The earlier copy `analysis_data/figure2_1000_zscores/` is kept as a record.
