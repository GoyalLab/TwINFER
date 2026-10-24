# `twinfer`

The TwINFER Python package: stochastic simulation of gene regulatory networks with twin (sister-cell) pairs, the covariance decomposition that separates regulation from cell-state heterogeneity, and network plots. The figure code in `paper_analysis/` and the `tutorials/` import from this package; the package never imports from them.

## Installation

From the repository root, in a Python ≥ 3.10 environment:

```bash
pip install -e package/
```

`-e` (editable) links `package/twinfer/` into the environment, so changes to the code take effect without reinstalling. The dependencies are listed in `pyproject.toml`. Check with `python -c "import twinfer; print(twinfer.__file__)"`.

## Modules

### `twinfer.inference.covariance_decomposition_z_scores`: inference

| Function | Does |
|---|---|
| `infer_with_twinfer(table, genes, time_points=None, ...)` | Scores gene pairs. Picks the data scenario from the table: **single time point** (twin pairs at one time → `z_rho`, `z_rho_s`, `z_rho_delta`, `inference_scenario`), **multiple time points** (adds the cross-time direction: `z_dag_xy`, `z_dag_yx`, `delta`, `p_delta`), **clones** (real data: one row per cell with `clone_id`, clones of any size, singletons allowed; optional `regulators=`, `targets=`). `correction="auto"` (default) uses Benjamini-Hochberg for clones (columns `q_rho`, `q_rho_delta`) and the fixed thresholds for twin-pair tables. |
| `classify_inference_scenario(z_rho, z_rho_s, z_rho_delta, thresholds=(1.96, 2.326, 1.96), correction="none", fdr=0.05)` | Inference scenario 1-4 (nothing, regulation, regulation + heterogeneity, heterogeneity), tested in the order z_rho, z_rho_s, z_rho_delta. `correction="bh"` tests z_rho with Benjamini-Hochberg over all pairs given and z_rho_delta with Benjamini-Hochberg over the pairs whose z_rho is significant (q ≤ `fdr`); z_rho_s keeps its threshold. |
| `q_values_of_z_rho(z_rho)`, `q_values_of_z_rho_delta(z_rho_delta, z_rho)`, `benjamini_hochberg(p)` | BH q-values (two-sided normal p-values; NaN is not counted as a test). |
| `twin_stats(x, y)` | All statistics of one gene pair from two arrays of shape (N, 2) (sister 1, sister 2). |
| `sister_arrays(table, genes, time_step)` | Turns the long twin table into those (N, 2) arrays. |
| `clone_twin_stats(x, y, clone)`, `clone_twin_stats_matrix(X, Y, clone)` | The clone-size-aware statistics for one pair or for all regulator × target pairs (real data). |

The module docstring gives every formula. Input tables:

* twin pairs: columns `clone_id`, `replicate` (1 or 2 = which sister), `time_step`, one column per gene;
* real data: one row per cell, `clone_id` and one column per gene (normalised values, e.g. log1p counts per 10,000), no `replicate` column.

### `twinfer.plotting.network_plots`: network figures

| Function | Does |
|---|---|
| `network_from_scenarios(result, ...)` | Turns the output of `infer_with_twinfer` into the inputs of `plot_grn`: scenarios 2-3 solid edges, scenario 4 dashed edges without direction, direction from `p_delta` when two time points were used. Returns `(plot_kwargs, edges)`. |
| `plot_grn(matrix, node_labels=None, ..., edge_styles=None, undirected_edges=None)` | Draws a signed, weighted directed network (`matrix[i, j]` = effect of node i on node j: positive = arrow, negative = bar). `edge_styles` sets a line style per edge, `undirected_edges` draws plain lines. |

### `twinfer.simulation`: stochastic simulation

| Module | Simulates |
|---|---|
| `gillespie_simulations` | Gene expression on a regulatory network (promoter switching, mRNA, protein; Gillespie algorithm), a burn-in, cell division into twin pairs, and the twins' expression over time. Run as `python -m twinfer.simulation.gillespie_simulations --help` or call `process_param_set`. Inputs: a connectivity matrix (`matrix[i, j]` = effect of gene i on gene j) and a parameter sheet, see `simulation_example_input_data/`. |
| `gillespie_drift` | The same with promoter activation rates (k_on) that ramp linearly after division, e.g. two halves of the clones drifting into a low and a high state. |
| `gillespie_pulse` | A transient pulse in k_on of chosen genes after division. |
| `gillespie_oscillations` | mRNA production modulated by an oscillation shared within a cell (for example a cell-cycle-like phase). |
| `shared_functions` | Helpers shared by the engines (matrix input, parameter assignment). |

### `twinfer.utils`

`paths` (repository and project folders, read from `TWINFER_PROJECT_ROOT`), `json_utils` (JSON export of numpy objects).

### Other modules

`twinfer.inference.infer`, `correlation_functions`, `network_naming`, `ranked_edges` and `twinfer.scoring` are the earlier, permutation-based pipeline and benchmark scoring helpers. They are kept for analyses that still import them; new code should use `covariance_decomposition_z_scores`.

## Environments

* `environment.yml`: conda environment used for development (simulation, inference, plotting).
* `grnboost_env.yml`: separate environment for the GRNBoost2 comparison (older numpy / pandas / scipy pins).
