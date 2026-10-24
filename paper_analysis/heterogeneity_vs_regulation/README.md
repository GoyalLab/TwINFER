# heterogeneity_vs_regulation

Two genes, A and B, simulated with and without regulation (A,B: no link; A → B: A activates B) in six scenarios:

| scenario | cells |
|---|---|
| A,B / A → B, single state | one population |
| A,B / A → B, 2 states | 3,000 twin pairs from a low-k_on population + 3,000 from a high-k_on population |
| A,B / A → B, drift | one population whose two halves drift to a low and a high k_on after division |

For each scenario, 100 replicates with 6,000 twin pairs are scored 20 hours after division with `z_rho`, `z_rho_s` and `z_rho_delta` (`twinfer.inference.covariance_decomposition_z_scores`). With 6,000 twin pairs, `z_rho` is significant in every replicate of every scenario except A,B single state (4 %); `z_rho_s` only in the scenarios with a shared state (2 states, drift: 100 %; single state: 0-3 %); and `z_rho_delta` detects regulation whether or not the cells have several states (A → B: 86-96 % of replicates; A,B: 4-9 %). A second set of plots shows how the z-scores change with the number of twin pairs (250 to 6,000).

## Code

| Folder | Script | Does |
|---|---|---|
| `simulation_code/` | `simulation_configs.py` | settings of the six base configurations (single state, low k_on, high k_on; for A,B and A → B) |
| | `simulate_heterogeneity_and_regulation.py` | simulates replicates of one configuration |
| | `run_generate_replicates_0_99.sbatch`, `run_generate_replicates_100_999.sbatch` | SLURM arrays: 1,000 replicates per configuration |
| | `simulate_drift_lowhigh_Kramp.py`, `run_simulate_drift_lowhigh_Kramp.sh` | the drift scenarios |
| | `README.md` | all simulation settings |
| `analysis_code/` | `twin_covariance_decomposition_z_scores.py` | z-scores per scenario × replicate × number of twin pairs → `analysis_data/twin_covariance_decomposition_z_scores.csv` |
| `plot_code/` | `plot_z_scores_6000_twins.py` | `z_rho_6000_twins.pdf`, `z_rho_s_6000_twins.pdf`, `z_rho_delta_6000_twins.pdf` |
| | `plot_n_twins_effect.py` | `z_rho_effect_n_twin.pdf`, `z_rho_s_effect_n_twin.pdf`, `z_rho_delta_effect_n_twin.pdf` |

## Run

Data are read from and written to `$TWINFER_PROJECT_ROOT/final_figure_data/heterogeneity_vs_regulation/` (`input_folder/`, `simulation_data/`, `analysis_data/`); plots go to `$TWINFER_PROJECT_ROOT/final_plots/heterogeneity_vs_regulation/`. Every script accepts other paths as arguments (see its header).

```bash
export TWINFER_PROJECT_ROOT=/path/to/project
export TWINFER_PYTHON=$(which python)          # the environment with twinfer installed
source env.sh                                  # from the repository root; used by the SLURM launchers

# 1. simulations (SLURM arrays)
sbatch paper_analysis/heterogeneity_vs_regulation/simulation_code/run_generate_replicates_0_99.sbatch
sbatch paper_analysis/heterogeneity_vs_regulation/simulation_code/run_simulate_drift_lowhigh_Kramp.sh

# 2. z-scores (about 35 s for 100 replicates with 4 processes)
python paper_analysis/heterogeneity_vs_regulation/analysis_code/twin_covariance_decomposition_z_scores.py

# 3. plots
python paper_analysis/heterogeneity_vs_regulation/plot_code/plot_z_scores_6000_twins.py
python paper_analysis/heterogeneity_vs_regulation/plot_code/plot_n_twins_effect.py
```

The analysis uses replicates 0-99 by default (`--reps`); `run_generate_replicates_100_999.sbatch` makes the remaining replicates of the 1,000-replicate set.
