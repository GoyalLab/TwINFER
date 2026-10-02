# Superseded LARRY twinscore sbatch scripts (2026-09-30)

Replaced by `paper_analysis/larry_hematopoiesis_validation/run_twinscore_larry.sbatch <run-name>` + `twinscore_runs/<run-name>.env` (checked by tests/smoke/smoke_larry_twinscore_runs.py). All 15 had identical resources (1 h, 32G, 1 cpu, partition short, account p32655) and differed only in GENE_SETS / SUMMARY_SUFFIX.

| Original | New |
|---|---|
| run_twinscore_correlation_high_v2.sbatch | `sbatch --job-name=twinscore_correlation_high_v2 run_twinscore_larry.sbatch correlation_high_v2` |
| run_twinscore_correlation_low.sbatch | `sbatch --job-name=twinscore_correlation_low run_twinscore_larry.sbatch correlation_low` |
| run_twinscore_correlation_low_par.sbatch | `sbatch --job-name=twinscore_correlation_low_par run_twinscore_larry.sbatch correlation_low_par` |
| run_twinscore_correlation_low_v2.sbatch | `sbatch --job-name=twinscore_correlation_low_v2 run_twinscore_larry.sbatch correlation_low_v2` |
| run_twinscore_correlation_mid_v2.sbatch | `sbatch --job-name=twinscore_correlation_mid_v2 run_twinscore_larry.sbatch correlation_mid_v2` |
| run_twinscore_supplement_mid.sbatch | `sbatch --job-name=twinscore_supplement_mid run_twinscore_larry.sbatch supplement_mid` |
| run_twinscore_variability_high.sbatch | `sbatch --job-name=twinscore_variability_high run_twinscore_larry.sbatch variability_high` |
| run_twinscore_variability_high_par.sbatch | `sbatch --job-name=twinscore_variability_high_par run_twinscore_larry.sbatch variability_high_par` |
| run_twinscore_variability_high_v2.sbatch | `sbatch --job-name=twinscore_variability_high_v2 run_twinscore_larry.sbatch variability_high_v2` |
| run_twinscore_variability_low.sbatch | `sbatch --job-name=twinscore_variability_low run_twinscore_larry.sbatch variability_low` |
| run_twinscore_variability_low_par.sbatch | `sbatch --job-name=twinscore_variability_low_par run_twinscore_larry.sbatch variability_low_par` |
| run_twinscore_variability_low_v2.sbatch | `sbatch --job-name=twinscore_variability_low_v2 run_twinscore_larry.sbatch variability_low_v2` |
| run_twinscore_variability_mid.sbatch | `sbatch --job-name=twinscore_variability_mid run_twinscore_larry.sbatch variability_mid` |
| run_twinscore_variability_mid_par.sbatch | `sbatch --job-name=twinscore_variability_mid_par run_twinscore_larry.sbatch variability_mid_par` |
| run_twinscore_variability_mid_v2.sbatch | `sbatch --job-name=twinscore_variability_mid_v2 run_twinscore_larry.sbatch variability_mid_v2` |
