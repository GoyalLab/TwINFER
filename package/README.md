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
| `infer_with_twinfer(table, genes, time_points=None, ...)` | Scores gene pairs. Picks the data scenario from the table: **single time point** (twin pairs at one time → `z_rho`, `z_rho_s`, `z_rho_Delta`, `inference_scenario`), **multiple time points** (adds the cross-time direction: `z_dag_xy`, `z_dag_yx`, `delta` = \|rho_dag_xy\| − \|rho_dag_yx\|, its permutation p-value `p_delta`; `cross_correlation_pairs="all" | "significant_rho" | "regulation"` limits it to all pairs, pairs with significant z_rho, or pairs in scenario 2 or 3), **clones** (real data: one row per cell with `clone_id`, clones of any size, singletons allowed; optional `regulators=`, `targets=`). `correction="auto"` (default) uses Benjamini-Hochberg for clones (columns `q_rho`, `q_rho_Delta`) and the fixed thresholds for twin-pair tables. |
| `save_result(table, path, settings=None)` | Writes a result table (`.csv`, `.csv.gz`, `.tsv`, `.tsv.gz` or `.parquet`, from the extension). `infer_with_twinfer(..., save_to=path)` uses it and also writes the settings of the call to `<name>_settings.json`. |
| `classify_inference_scenario(z_rho, z_rho_s, z_rho_Delta, thresholds=(1.96, 2.326, 1.96), correction="none", fdr=0.05)` | Inference scenario 1-4 (nothing, regulation, regulation + heterogeneity, heterogeneity), tested in the order z_rho, z_rho_s, z_rho_Delta. `correction="bh"` tests z_rho with Benjamini-Hochberg over all pairs given and z_rho_Delta with Benjamini-Hochberg over the pairs whose z_rho is significant (q ≤ `fdr`); z_rho_s keeps its threshold. |
| `q_values_of_z_rho(z_rho)`, `q_values_of_z_rho_Delta(z_rho_Delta, z_rho)`, `benjamini_hochberg(p)` | BH q-values (two-sided normal p-values; NaN is not counted as a test). |
| `twin_stats(x, y)` | All statistics of one gene pair from two arrays of shape (N, 2) (sister 1, sister 2). |
| `sister_arrays(table, genes, time_step)` | Turns the long twin table into those (N, 2) arrays. |
| `clone_twin_stats(x, y, clone)`, `clone_twin_stats_matrix(X, Y, clone)` | The clone-size-aware statistics for one pair or for all regulator × target pairs (real data). |

### `twinfer.inference.ranked_edges`: ranked lists

`ranked_edge_lists(result, scores=None, top=None, direction_alpha=0.05)` ranks the gene pairs of an `infer_with_twinfer` result by |z_rho_s| (shared state) and by |z_rho_Delta| (regulation), among the pairs whose z_rho is significant (q_rho ≤ 0.05 when present, else \|z_rho\| > 1.96). Returns one table per score with a `rank` column. When the result has two time points it adds a third list, `z_dag`: one row per direction found by `pair_directions(z_dag_xy, z_dag_yx, delta, p_delta, scenario)`, source → target, ranked by \|z_dag\| of that direction. In scenario 2 (regulation) each direction is tested on its own (\|z_dag\| > 1.96), so mutual regulation gives two rows; in scenario 3 (regulation + a shared state, which correlates both directions) one direction, the sign of `delta`, when `p_delta` < 0.05; otherwise one row with `directed` False. `network_from_scenarios` draws the same directions.

The module docstring gives every formula. Input tables:

* twin pairs: columns `clone_id`, `replicate` (1 or 2 = which sister), `time_step`, one column per gene;
* real data: one row per cell, `clone_id` and one column per gene (normalised values, e.g. log1p counts per 10,000), no `replicate` column.

### `twinfer.plotting.network_plots`: network figures

| Function | Does |
|---|---|
| `network_from_scenarios(result, ...)` | Turns the output of `infer_with_twinfer` into the inputs of `plot_grn`: scenarios 2-3 solid edges, scenario 4 dashed edges without direction, directions by `pair_directions` when two time points were used (both directions possible in scenario 2), otherwise no direction; edge colour = `weight` (default `z_dag` of the drawn direction with two time points, `z_rho_Delta` with one); a directed edge is an arrow or a bar by the sign of its `z_dag`. Returns `(plot_kwargs, edges)`. |
| `plot_grn(matrix, node_labels=None, ..., edge_styles=None, undirected_edges=None)` | Draws a signed, weighted directed network (`matrix[i, j]` = effect of node i on node j: positive = arrow, negative = bar). `edge_styles` sets a line style per edge, `undirected_edges` draws plain lines, `edge_signs` sets arrow or bar apart from the colour. |

### `twinfer.simulation`: stochastic simulation

| Module | Simulates |
|---|---|
| `gillespie_simulations` | Gene expression on a regulatory network (promoter switching, mRNA, protein; Gillespie algorithm), a burn-in, cell division into twin pairs, and the twins' expression over time. Run as `python -m twinfer.simulation.gillespie_simulations --help` or call `process_param_set`. Inputs: a connectivity matrix (`matrix[i, j]` = effect of gene i on gene j) and a parameter sheet, see `simulation_example_input_data/`. |
| `gillespie_drift` | The same with promoter activation rates (k_on) that ramp linearly after division, e.g. two halves of the clones drifting into a low and a high state. |
| `shared_functions` | Helpers shared by the engines (matrix input, parameter assignment). |

### `twinfer.utils`

`paths` (repository and project folders, read from `TWINFER_PROJECT_ROOT`), `json_utils` (JSON export of numpy objects).

### Other modules

`twinfer.inference.network_naming` converts networks with real gene names to the `gene_1, gene_2, ...` names of the simulator and back.

## Environments

* `environment.yml`: conda environment used for development (simulation, inference, plotting).
* `grnboost_env.yml`: separate environment for the GRNBoost2 comparison (older numpy / pandas / scipy pins).
