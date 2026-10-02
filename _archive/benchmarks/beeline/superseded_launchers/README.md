# Superseded Beeline launchers (2026-09-30)

Replaced by `benchmarks/beeline/run_beeline.sh <run-name>` + `benchmarks/beeline/runs/<run-name>.env`. Equivalence (same commands, redirects, modules, files, per-dataset sub-configs) is checked by `tests/smoke/smoke_beeline_launchers.py`. SBATCH resources of each original are kept in the SBATCH_HINT line of its .env.

| Original | New command |
|---|---|
| run_autoregulation_genie3_grnboost2.sh | `bash benchmarks/beeline/run_beeline.sh autoregulation_genie3_grnboost2` |
| run_e13_pos100_benchmark.sh | `bash benchmarks/beeline/run_beeline.sh e13_pos100_benchmark` |
| run_e13_pos100_benchmark_t1_10.sh | `bash benchmarks/beeline/run_beeline.sh e13_pos100_benchmark_t1_10` |
| run_e13_pos100_genie3_grnboost2.sh | `bash benchmarks/beeline/run_beeline.sh e13_pos100_genie3_grnboost2` |
| run_e13_pos100_genie3_grnboost2_t1_10.sh | `bash benchmarks/beeline/run_beeline.sh e13_pos100_genie3_grnboost2_t1_10` |
| run_genie3_grnboost2_sequential.sh | `bash benchmarks/beeline/run_beeline.sh genie3_grnboost2_sequential` |
| run_mixed_network_sweep_benchmark.sh | `bash benchmarks/beeline/run_beeline.sh mixed_network_sweep_benchmark` |
| run_mixed_network_sweep_benchmark_t1_10.sh | `bash benchmarks/beeline/run_beeline.sh mixed_network_sweep_benchmark_t1_10` |
| run_mixed_network_sweep_genie3_grnboost2.sh | `bash benchmarks/beeline/run_beeline.sh mixed_network_sweep_genie3_grnboost2` |
| run_mixed_network_sweep_genie3_grnboost2_t1_10.sh | `bash benchmarks/beeline/run_beeline.sh mixed_network_sweep_genie3_grnboost2_t1_10` |
| run_networksweep_benchmark.sh | `bash benchmarks/beeline/run_beeline.sh networksweep_benchmark` |
| run_networksweep_benchmark_t1_10.sh | `bash benchmarks/beeline/run_beeline.sh networksweep_benchmark_t1_10` |
| run_networksweep_genie3_grnboost2_t1_10.sh | `bash benchmarks/beeline/run_beeline.sh networksweep_genie3_grnboost2_t1_10` |
| run_real_network_benchmark.sh | `bash benchmarks/beeline/run_beeline.sh real_network_benchmark` |
| run_real_network_benchmark_t1_10.sh | `bash benchmarks/beeline/run_beeline.sh real_network_benchmark_t1_10` |
| run_real_network_genie3_grnboost2_t1_10.sh | `bash benchmarks/beeline/run_beeline.sh real_network_genie3_grnboost2_t1_10` |
