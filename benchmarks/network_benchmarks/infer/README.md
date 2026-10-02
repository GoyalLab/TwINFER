# infer

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| boolode_inference_twinfer.ipynb | Compare twin-pair gene-gene correlations at t1 vs t2 |
| boolode_real_infer_allpairs.py | ALL_PAIRS TwINFER inference on the BoolODE real-network twin sims (B_cell_activation, EMT_real, Pluripotent_real), converted by boolode_real |
| infer_cyclic_g3456.py | Reruns TwINFER inference (with fan-out separation) on the cyclic_g3/g4/g5/g6 raw simulations, mirroring benchmark_network_sweep.ipynb's Part |
| infer_e13_pos100.py | TwINFER inference over the 3 x 10 e13_pos100 simulations at simulation_data/network_sweep_final/e13_pos100/ -- these are 3 of the network_sw |
| infer_e13_pos100_allpairs.py | ALL_PAIRS unrestricted rerun of infer_e13_pos100.py: same 30 simulations, same everything, except alpha_gene_gene_corr/alpha_stage3 -> 0.999 |
| infer_e13_pos100_allpairs_t1_10.py | ALL_PAIRS unrestricted rerun of infer_e13_pos100.py: same 30 simulations, same everything, except alpha_gene_gene_corr/alpha_stage3 -> 0.999 |
| infer_e13_pos100_alpha05.py | One-off rerun of infer_e13_pos100.py with alpha_gene_gene_corr=0.05 (instead of the package default 0.01), to test whether loosening Stage 1 |
| infer_mixed_network_sweep.py | TwINFER inference over every finished simulation in simulation_data/mixed_network_sweep/, mirroring rerun_twinfer_150.py's run_one() (same i |
| infer_mixed_network_sweep_allpairs.py | ALL_PAIRS unrestricted rerun of infer_mixed_network_sweep.py: same simulations, same everything, except alpha_gene_gene_corr/alpha_stage3 -> |
| infer_mixed_network_sweep_allpairs_t1_10.py | ALL_PAIRS unrestricted rerun of infer_mixed_network_sweep.py: same simulations, same everything, except alpha_gene_gene_corr/alpha_stage3 -> |
| infer_network_simulation_boolode_sims.py | BEHAVIOUR CHANGES vs the pre-reorg runs: (1) Step 1 now uses alpha_gene_gene_corr (default 0.01) with a clone/analytic null, not use_scrambl |
| infer_network_simulation_cyclic.py | BEHAVIOUR CHANGES vs the pre-reorg runs: (1) Step 1 now uses alpha_gene_gene_corr (default 0.01) with a clone/analytic null, not use_scrambl |
| infer_network_simulation_multistate.py | TwINFER inference (infer_with_twinfer) over the new multi-state real-network simulations: the IC-unset ("multistate") and seeded-IC ("seeded |
| infer_network_simulation_multistate_nofilter.py | TwINFER inference (infer_with_twinfer) over the new multi-state real-network simulations: the IC-unset ("multistate") and seeded-IC ("seeded |
| infer_network_simulation_network_sweep.py | BEHAVIOUR CHANGES vs the pre-reorg runs: (1) Step 1 now uses alpha_gene_gene_corr (default 0.01) with a clone/analytic null, not use_scrambl |
| infer_network_simulation_real_network.py | Full TwINFER inference over every real-network simulation replicate. |
| infer_network_simulation_real_network_nofilter.py | Full TwINFER inference over every real-network simulation replicate, with every COVERAGE-controlling significance/threshold gate in the pipe |
| infer_network_sweep_final_allpairs.py | ALL_PAIRS unrestricted rerun for the REST of network_sweep_final's OFAT topology sweep -- e5_pos100_density, e9_pos0_sign_ratio, e9_pos50_si |
| infer_network_sweep_final_allpairs_t1_10.py | ALL_PAIRS unrestricted rerun for the REST of network_sweep_final's OFAT topology sweep -- e5_pos100_density, e9_pos0_sign_ratio, e9_pos50_si |
| infer_real_network_allpairs.py | ALL_PAIRS unrestricted rerun of infer_network_simulation_real_network.py: same 65 replicates across 7 networks (GSD/HSC/VSC/mCAD/B_cell_acti |
| infer_real_network_allpairs_t1_10.py | ALL_PAIRS unrestricted rerun of infer_network_simulation_real_network.py: same 65 replicates across 7 networks (GSD/HSC/VSC/mCAD/B_cell_acti |
| infer_real_networks_extra.py | Modernized TwINFER-inference JSON generator for benchmark_twinfer_vs_boolode.ipynb's TwINFER-simulation track. |
| network_sweep_twinfer_new.py | BEHAVIOUR CHANGES vs the pre-reorg runs: (1) Step 1 now uses alpha_gene_gene_corr (default 0.01) with a clone/analytic null, not use_scrambl |
| rerun_twinfer_150.py | Full, clean TwINFER rerun against the finalized, SCODE-anchor-verified 150-file manifest (source_manifest_150.json) -- the exact same source |
| resume_infer_real_networks_extra.py | Resume pass for infer_real_networks_extra.py: only (re)runs tasks whose fanout_on/fanout_off JSON pair isn't already a *regenerated* (curren |
| run_boolode_real_infer_allpairs.sh | ALL_PAIRS TwINFER inference on the BoolODE twin sims for the networks given as arguments, all their |
| run_e13_pos100_allpairs_t1_10.sh | t1=10,t2=20 rerun of infer_e13_pos100_allpairs.py (measure metrics farther from the division |
| run_infer_all_nofilter.sh | mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs   # [2026-09-30 replaced by env.sh variable] |
| run_infer_mixed_network_sweep.sh | TwINFER inference over every finished mixed_network_sweep simulation |
| run_infer_multistate.sh | mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate/logs   # [2026-09-30 |
| run_infer_multistate_nofilter.sh | mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate_nofilter/logs   # [2 |
| run_infer_network.sh | mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference/logs   # [2026-09-30 replaced b |
| run_infer_network_nofilter.sh | mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter/logs   # [2026-09-30 r |
| run_mixed_network_sweep_allpairs.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_mixed_network_sweep_allpairs_t1_10.sh | t1=10,t2=20 rerun of infer_mixed_network_sweep_allpairs.py (measure metrics farther from the |
| run_network_sweep_final_allpairs.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_network_sweep_final_allpairs_t1_10.sh | t1=10,t2=20 rerun of infer_network_sweep_final_allpairs.py (measure metrics farther from the |
| run_real_network_allpairs.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_real_network_allpairs_t1_10.sh | t1=10,t2=20 rerun of infer_real_network_allpairs.py (measure metrics farther from the division |
| run_resume_infer_real_networks.sh | cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis   # [2026-09-30 replaced by env.sh variable] |
| run_twinfer_150_rerun.sh | Full TwINFER rerun against the same 150-file manifest used to build the |
| run_twinfer_autoreg_rerun.sh | Full TwINFER rerun for the 40-file autoregulation manifest (4 topologies x |
