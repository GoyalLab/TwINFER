# scripts

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| cleanup_cyclic_autoreg.py | Compress the before_division checkpoint files for cyclic (g3/g4/g5/cyclic_6_nodes) and autoregulation -- unlike network_sweep_final, every o |
| cleanup_network_sweep_final.py | Cleanup for network_sweep_final's simulation_data directory, per the finalized 150-file manifest (source_manifest_150.json): |
| generate_cyclic_g3456_configs.py | Generates one BLRunner config per dataset for the cyclic_g3/g4/g5/g6 datasets (simple directed cycles, 3/4/5/6 genes) -- same 7 enabled algo |
| generate_e13_pos100_configs.py | Generate BEELINE configs for the e13_pos100 benchmark (3 datasets: n6_e13_pos100_rep{1,2,3}), same algorithm set/layout as generate_mixed_ne |
| generate_emt_pluripotent_configs.py | Generates one BLRunner config per (dataset, run) pair for the EMT/Pluripotent real-network datasets -- same 7 enabled algorithms (PIDC, GENI |
| generate_mixed_network_sweep_configs.py | Generate BEELINE configs for the mixed_network_sweep benchmark, mirroring config-files/config_networksweep_final_20260824.yaml: |
| generate_multistate_configs.py | Generates one BLRunner config per (dataset, run) pair for the new multi-state real-network datasets (GSD/HSC/EMT IC-unset "multistate" + GSD |
| generate_t1_10_beeline_configs.py | 2026-09-18: BEELINE configs for the t1=10,t2=20 twin_paired-only reruns of all 4 simulated benchmark families (e13_pos100, network_sweep_fin |
| run_autoreg_pidc_fix.sh | SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable] |
| run_autoregulation_genie3_grnboost2.sh | GENIE3 + GRNBOOST2, all 4 autoregulation datasets, ONE BLRunner.py process |
| run_autoregulation_safe.sh | PIDC, PPCOR, SCODE, SCSGL, PEARSON for the 4 autoregulation datasets |
| run_cleanup_cyclic_autoreg.sh | Compress before_division checkpoint files for cyclic (g3/g4/g5/cyclic_6_nodes) |
| run_cleanup_network_sweep_final.sh | Cleanup network_sweep_final's simulation_data per the finalized 150-file |
| run_e13_pos100_benchmark.sh | BEELINE benchmark for the e13_pos100 datasets (3 of the network_sweep_final |
| run_e13_pos100_benchmark_t1_10.sh | t1=10,t2=20 rerun of run_e13_pos100_benchmark.sh: PIDC/PPCOR/SCODE/SCSGL/PEARSON over the |
| run_e13_pos100_genie3_grnboost2.sh | GENIE3 + GRNBOOST2 for the 3 e13_pos100 datasets, single BLRunner.py process |
| run_e13_pos100_genie3_grnboost2_t1_10.sh | GENIE3 + GRNBOOST2 for the 3 e13_pos100 t1=10 datasets, single BLRunner.py process |
| run_genie3_e17rep2_finish.sh | Finishes the last 5 GENIE3 runs (grn_n6_e17_pos100_density_rep2, labels 5-9) |
| run_genie3_grnboost2_sequential.sh | GENIE3 + GRNBOOST2, all 150 runs, ONE BLRunner.py process (no per-dataset |
| run_genie3_ppcor_fix.sh | Fixes two source-file problems found by cross-checking every algorithm's |
| run_mixed_network_sweep_benchmark.sh | BEELINE benchmark for the mixed_network_sweep datasets: PIDC, PPCOR, SCODE, |
| run_mixed_network_sweep_benchmark_t1_10.sh | t1=10,t2=20 rerun of run_mixed_network_sweep_benchmark.sh: PIDC/PPCOR/SCODE/SCSGL/PEARSON over |
| run_mixed_network_sweep_genie3_grnboost2.sh | GENIE3 + GRNBOOST2 for every mixed_network_sweep dataset, ONE BLRunner.py |
| run_mixed_network_sweep_genie3_grnboost2_t1_10.sh | GENIE3 + GRNBOOST2 for every mixed_network_sweep t1=10 dataset, ONE BLRunner.py process (no |
| run_networksweep_benchmark.sh | mkdir -p /home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/network_sweep/beeline_inference/logs   # [2026-09-30 replaced by env.sh variabl |
| run_networksweep_benchmark_t1_10.sh | t1=10,t2=20 rerun of run_networksweep_benchmark.sh (the broader network_sweep_final OFAT |
| run_networksweep_cycle.sh | mkdir -p /home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/network_sweep/beeline_inference/logs   # [2026-09-30 replaced by env.sh variabl |
| run_networksweep_final_full_rerun.sh | Full, clean rerun of 5 of the 7 BEELINE algorithms (PIDC, PPCOR, SCODE, |
| run_networksweep_genie3_grnboost2_t1_10.sh | GENIE3 + GRNBOOST2 for every network_sweep_final (broader OFAT sweep) t1=10 dataset, ONE |
| run_pidc_fix.sh | PIDC-only rerun, all 15 datasets, 15-way parallel -- fixes the missing |
| run_ppcor_rerun.sh | Reruns ONLY PPCOR, in place, after removing ppcorRunner.py's pre-filtering of |
| run_real_network_benchmark.sh | mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference/logs   # [2026-09-30 replaced b |
| run_real_network_benchmark_t1_10.sh | t1=10,t2=20 rerun of run_real_network_benchmark.sh (65 replicates / 7 networks): |
| run_real_network_genie3_grnboost2_t1_10.sh | GENIE3 + GRNBOOST2 for every real_networks t1=10 dataset, ONE BLRunner.py process (no |
| run_real_network_twinpaired_backfill.sh | Backfills the missing/incomplete twin_paired BEELINE runs for the curated |
| run_real_network_twinpaired_backfill_chunk.sh | Backfills the missing/incomplete twin_paired BEELINE runs for the curated |
| run_real_networks_chunk.sh | Runs all 7 enabled BEELINE algorithms (PIDC, GENIE3, GRNBOOST2, PPCOR, SCODE |
| run_scode_rep6.sh | Reruns ONLY SCODE, at nRep=6 (BEELINE's own reference default -- see |
| run_scode_rep6_chunk.sh | Processes one chunk-list file of per-run SCODE nRep=6 configs (one config per |
| run_scode_rep6_final_retry.sh | Final cleanup pass for the ~18 SCODE nRep=6 runs left over from |
| run_verify_scoring.sh | cd /home/gzu5140/TwINFER_KA/code/Beeline/_verify_scoring_work   # [2026-09-30 replaced by env.sh variable] |
