# drift_multiple_state

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| build_scenario78_upDrift_AB.py | Build two synthetic mixed scenarios for the 6-scenario z-score plot: |
| compare_recover_vs_multistate_dists.py | Compare gene-expression distributions: low/mid DRIFT (recover, 300 h) vs the static 2-state low/mid merge (multistate_lowbase = low_k_on + n |
| compute_abs_drift_zscores.py | NEW, additive statistic: z-score for the ABSOLUTE change in rho_Delta between t1 and t2, computed for BOTH the twin-based reference and the  |
| infer_drift_files.py | Run infer_with_twinfer (the real pipeline function) on the regenerated 2-state drift simulations and save the diagnostics. |
| inspect_g1_g2_crosscorr.py | Inspect the Step-4 cross-time relationship gene_1(t1) -> gene_2(t2) (and the reverse) for a single drift-simulation replicate, exactly as ru |
| plot_6scenario.py | Two clean figures for the 6-scenario x 20-replicate TwINFER analysis (gene_1-gene_2), from six_scenario_zscores.csv: |
| plot_6scenario_all_zscores.py | Boxplot EVERY z-score infer.py's Steps 1-4 produce for gene_1-gene_2, across replicates, one panel per z-score, one box per scenario. |
| plot_6scenario_newdrift.py | Same two figures + threshold-crossing heatmap as plot_6scenario.py, but with the `drift_A_to_B` scenario replaced by the freshly regenerated |
| plot_abs_drift_zscores.py | Boxplot the NEW abs-drift z-scores (compute_abs_drift_zscores.py) across replicates, one box per scenario, for all 8 scenarios currently on  |
| plot_drift_allsteps.py | Plot all five TwINFER z-scores (Step 1, Step 2 z_het, z_divergence -- each at t1 and t2 -- plus Step 3 z_d and Step 4 z both directions) for |
| plot_drift_infer.py | Plot infer_with_twinfer results on the regenerated 2-state drift sims (drift_infer_summary.csv): classification breakdown + the z-scores the |
| plot_drift_zscores.py | Plot TwINFER correlations and Step-1/2/3/4 z-scores for the 20 drift-simulation replicates, regulated (A_to_B) vs unregulated (A_B_no_reg),  |
| plot_half_subsample_comparison.py | Full-population (3 reps) vs. half-population (3 reps x 5 random clone-subsample draws each = 15 points) comparison, per scenario, for every  |
| plot_lowmid_100h_step12_vs_time.py | Low/mid drift (`recover` variant), 100 h: Step-1 and Step-2 z-scores + their correlations for gene_1-gene_2, one point per hour, mean of the |
| plot_lowmid_100h_step2_vs_time.py | For the 100 h "low/mid" drift (`recover` variant: one twin sub-population holds k_on at 0.12x, the other ramps 0.12x -> 1.0x baseline over 1 |
| plot_lowmid_combined.py | Overlay all four low->mid series on one figure: {K_frozen, K_ramp} x {A_B (unreg), A_to_B (reg)} from the aggregated <tag>_zscores_vs_time.c |
| plot_single2K_combined.py | Single-state (k_on = 0.66 median) comparison: what does a wrong Hill K do to the TwINFER z-scores when the population is NOT multi-state? |
| plot_state_ratios_vs_time.py | For the 2-state drift simulations, plot the ratio of the population-mean expression as a function of time, for gene_1/gene_2 x mRNA/protein: |
| regenerate_drift_A_B.py | Regenerate the A_B (no-regulation) 2-state drift simulation, 20 replicates. |
| regenerate_drift_A_B.sh | cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable] |
| regenerate_drift_A_B_p32655.sh | cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable] |
| regenerate_drift_A_to_B.py | Regenerate the A_to_B (regulation) 2-state drift simulation, 20 replicates. |
| regenerate_drift_A_to_B.sh | cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable] |
| regenerate_drift_lowmid.py | Rebuild of the low->mid 2-state drift, both "kinds", both networks. |
| regenerate_drift_variant.py | Generate the drift 2-state simulations for two NEW drift regimes, both networks: |
| regenerate_single_2K.py | Single-state A_B / A_to_B at the median k_on = 0.66, with the gene_1->gene_2 Hill K calibrated either at k_on = 0.66 (correct / native) or a |
| rename_A_B_reps.py | Renumber the regenerated A_B (no-reg) 2-state drift sims to a clean rep_1 .. rep_20 set. All A_B sims use identical params (rows [[0,0]]); t |
| run_6scenario_zscores.py | Run infer_with_twinfer FROM SCRATCH on 6 scenarios x 20 replicates and save all Step 1/2/3/4 metrics and the five z-scores for the gene_1-ge |
| run_6scenario_zscores.sh | cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable] |
| run_8scenario_drift_only.sh | Login-node run OOM'd (12 concurrent loky workers each loading a ~125MB/6000-cell |
| run_8scenario_zscores.sh | 8 scenarios x {20 reps baseline/multistate, 3 reps drift_lowmid} at two (t1,t2) |
| run_abs_drift_zscores.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_abs_drift_zscores_t10.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_drift_lowmid_100h.sh | "low/mid drift" = the recover variant (up: k_on 0.12x -> 1.0x baseline; down: stays 0.12x) |
| run_drift_lowmid_rebuild.sh | 4 array tasks = (kind x network). Each: (1) simulate low->mid, then (2) score |
| run_drift_lowmid_v2.sh | 5 array tasks: 2x K_ramp (live mean-field K, reuse K_frozen burn-ins) + 3x single-state. |
| run_drift_simulation.sh | ~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/drift_multiple_state/simulate_drift_multiple_s |
| run_drift_variants.sh | 4 array tasks: (network x variant) |
| run_infer_drift_replicates.py | TwINFER Step 1 / Step 2 / Step 3 z-scores for the 20 drift-simulation replicates, regulated (A_to_B) vs unregulated (A_B_no_reg) 2-state pop |
| run_infer_drift_replicates.sh | cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable] |
| run_infer_drift_replicates_10k.sh | cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable] |
| run_lowmid_100h_analysis.sh | mkdir -p /projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/lowmid_100h/logs   # [2026-09-30 replaced by env.sh variable] |
| run_lowmid_300h_all.sh | ONE job: (1) simulate the low/mid drift (recover variant) for 300 h twin time, |
| run_sametime_twin_crosscorr.py | IDEA TEST: same-timepoint twin cross-correlation. |
| score_lowmid_zscores_vs_time.py | All TwINFER z-scores + their correlations for the gene_1-gene_2 pair, as a function of twin time, for one (kind x network) group of the low- |
| simulate_drift_multiple_states.py | Script to generate multiple states before division and this is inherited by both daughter cells |
| subsample_half_and_score.py | Subsampling-noise study: for each of the 8 scenarios' existing 3 replicates, draw K independent random halves of the clone population (no re |
| visualize_drift_simulation.ipynb | Extended fig E8 |
