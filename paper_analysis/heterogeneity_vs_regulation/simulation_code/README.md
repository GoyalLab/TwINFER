# simulation_code: heterogeneity_vs_regulation

Simulations behind the six scenarios of this figure: two genes with no link (A,B) or with A activating B (A → B), in a single state, in two states, or drifting into two states after division. Outputs go to `final_figure_data/heterogeneity_vs_regulation/simulation_data/` under the project folder (`TWINFER_PROJECT_ROOT`).

## 1. Single-state and two-state scenarios (`simulation_data/single_and_two_state_simulations/`)

Six configurations, 1,000 replicates each: A_B, A_B_low_k_on, A_B_high_k_on, A_to_B, A_to_B_low_k_on, A_to_B_high_k_on. A single-state scenario is one A_B or A_to_B replicate; a two-state scenario combines a low_k_on and a high_k_on replicate of the same number (3,000 twin pairs of each, done in the analysis).

| File | Role |
|---|---|
| `simulation_configs.py` | Settings of the six configurations (`get_base_configs`) |
| `simulate_heterogeneity_and_regulation.py` | Simulates a range of replicates of one configuration (env `CONFIG_INDEX`, `REP_START`, `REP_COUNT`, `NCORES`, `OUT_DIR`, `INPUT_DIR`) |
| `run_generate_replicates_0_99.sbatch` | SLURM array: replicates 0-99 of all six configurations (600 tasks, one replicate each) |
| `run_generate_replicates_100_999.sbatch` | SLURM array: replicates 100-999 (5,400 tasks) |

Settings: 6,000 cells before division (= 6,000 twin pairs); 1,000 h burn-in; twins simulated for 48 h after division and sampled hourly; additive model for multiple regulators. Gene parameters from `median_parameter.csv`: single state rows 0 and 1 (k_on 0.66), low_k_on row 2 (k_on 0.12), high_k_on row 3 (k_on 1.66).

## 2. Drift scenarios (`simulation_data/drift_lowhigh_Kramp/`, networks A_B and A_to_B)

1,000 replicates per network.

| File | Role |
|---|---|
| `simulate_drift_lowhigh_Kramp.py` | Drift simulation (uses `twinfer.simulation.gillespie_drift`) |
| `run_simulate_drift_lowhigh_Kramp.sh` | SLURM launcher: 3 array tasks (A_B replicates 0-999; A_to_B replicates 0-499 and 500-999) |

Settings: replicate n starts from the burn-in end state of replicate n of the single-state set above, so both sets share the same mother cells. At division the 6,000 mother cells are split in half, and both sisters of a clone follow the same path: in clones 0-2999 k_on of both genes ramps from 0.66 to 0.12, in clones 3000-5999 from 0.66 to 1.66, linearly over 10 h after division. For A_to_B, the activation threshold (Hill K) of gene 1 → gene 2 follows the current mean-field value of gene 1 (updated every 0.05 h). Twins are simulated for 48 h after division and sampled hourly; each output folder has a `log.jsonl` with the per-replicate settings.

Command used: `python simulate_drift_lowhigh_Kramp.py --network <A_B|A_to_B> --out-dir <OUT> --reps <range> --jobs 12 --cores-per 4 --twin-time 48 --tau 10 --burnin-hours 1000 --dt-K 0.05`.

## Inputs

`final_figure_data/heterogeneity_vs_regulation/input_folder/` (override with `INPUT_DIR`): the connectivity matrices `connectivity_matrix_2_gene_no_interaction.txt`, `connectivity_matrix_A_B.txt`, `connectivity_matrix_A_to_B.txt` and the parameter sheet `median_parameter.csv`. Rows 0-3 of `median_parameter.csv` are the same as in `simulation_example_input_data/median_parameter.csv` of this repository.

## Running

```bash
export TWINFER_PROJECT_ROOT=/path/to/project
source env.sh                                   # from the repository root
sbatch paper_analysis/heterogeneity_vs_regulation/simulation_code/run_generate_replicates_0_99.sbatch
sbatch paper_analysis/heterogeneity_vs_regulation/simulation_code/run_generate_replicates_100_999.sbatch
sbatch paper_analysis/heterogeneity_vs_regulation/simulation_code/run_simulate_drift_lowhigh_Kramp.sh   # needs the single-state burn-in states
```
