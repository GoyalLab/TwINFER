# twinfer.simulation

Stochastic simulation (Gillespie algorithm) of gene expression on a regulatory network. Each gene has a promoter that switches on and off, mRNA and protein; regulators change the promoter activation rate of their targets. A population of cells is simulated to steady state (burn-in), every cell divides into two sisters (twins), and the twins are followed after division. By default both sisters start from an exact copy of their mother's state (promoters and molecule counts); with `divide_binomial` the molecules are split between the sisters binomially.

| Module | Simulates |
|---|---|
| `gillespie_simulations.py` | The standard simulation. Run from the command line (`python -m twinfer.simulation.gillespie_simulations --help`) or call `process_param_set`. |
| `gillespie_drift.py` | The same, with promoter activation rates (k_on) that ramp linearly after division, e.g. two halves of the clones drifting into a low and a high state. |
| `shared_functions.py` | Helpers shared by the engines (reading the connectivity matrix, assigning parameters to genes). |

**Inputs**

* a connectivity matrix: a text file with one row and one column per gene, `matrix[i, j]` = effect of gene i on gene j (positive = activation, negative = repression, 0 = no link);
* a parameter sheet: one row per gene with the promoter switching rates, production rates and half-lives of mRNA and protein, and the regulation parameters.

Examples of both are in [`simulation_example_input_data/`](../../../simulation_example_input_data/).

**Output**: a table with one row per cell and time point: `cell_id`, `time_step` (hours after division), `<gene>_mRNA`, `<gene>_protein`, `replicate` (1 or 2 = which sister), `clone_id` (the twin pair). This is the input of `twinfer.inference.covariance_decomposition_z_scores.infer_with_twinfer`. An example is in [`simulation_example_output_data/`](../../../simulation_example_output_data/).
