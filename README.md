# TwINFER

TwINFER tells apart two reasons why two genes are correlated across single cells: **regulation** (one gene acts on the other inside each cell) and **cell-state heterogeneity** (the cells sit in different states that set both genes). It uses lineage tracing: sister cells ("twins") inherit the same state from their mother cell, so a correlation that is shared by twins points to heterogeneity, and a correlation of the twin *differences* points to regulation.

This repository contains the `twinfer` Python package, tutorials, and the code behind the figures of the paper.

## How it works

For a pair of genes X and Y measured in N twin pairs, TwINFER splits the expression correlation into a part shared by twins and a part that differs between twins, and reports three analytic z-scores:

| z-score | measures |
|---|---|
| `z_rho` | the correlation of X and Y over all cells (Spearman) |
| `z_rho_s` | the twin-shared part: X in one sister against Y in the other sister |
| `z_rho_delta` | the correlation of the twin differences of X and Y |

Each gene pair is then assigned an inference scenario:

| scenario | z_rho | z_rho_s | z_rho_delta |
|---|---|---|---|
| 1 nothing | not significant | | |
| 2 regulation | significant | not significant | |
| 3 regulation + heterogeneity | significant | significant | significant |
| 4 heterogeneity | significant | significant | not significant |

The z-scores are tested in this order: `z_rho`, then `z_rho_s`, then `z_rho_delta`. Significant means \|z\| > 1.96 for `z_rho` and `z_rho_delta`, > 2.326 for `z_rho_s` (the thresholds can be changed). When many gene pairs are scored at once, as in real data, `z_rho` and `z_rho_delta` are instead tested with a Benjamini-Hochberg correction: `z_rho` over all pairs, `z_rho_delta` over the pairs whose `z_rho` is significant (q ≤ 0.05 for both).

When twins are measured at two time points, the cross-time correlation between sisters (X in one sister earlier against Y in the other sister later, and the reverse) gives the direction of regulation.

The same statistics work for lineage-tracing data with clones of any size, including singletons.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/GoyalLab/TwINFER.git
cd TwINFER
conda create -n twinfer python=3.12
conda activate twinfer
pip install -e package/          # installs twinfer and its dependencies
pip install jupyterlab           # only to run the tutorials
```

Check the installation:

```bash
python -c "import twinfer; print(twinfer.__file__)"
```

## Quick start

```python
import pandas as pd
from twinfer.inference.covariance_decomposition_z_scores import infer_with_twinfer
from twinfer.plotting.network_plots import network_from_scenarios, plot_grn

# Twin pairs: one row per cell and time point, columns clone_id, replicate (1 or 2 = which sister), time_step, one column per gene
table = pd.read_csv("simulation_example_output_data/df_rows_0_0_20112025_222419_ncells_6000_A_to_B_4e702629.csv")
genes = ["gene_1_mRNA", "gene_2_mRNA"]

result = infer_with_twinfer(table, genes, time_points=[20])          # z_rho, z_rho_s, z_rho_delta, inference_scenario per gene pair
result = infer_with_twinfer(table, genes, time_points=[1, 20])       # adds the direction (z_dag_xy, z_dag_yx, delta, p_delta)

# Real data: one row per cell with clone_id and normalised expression (no replicate column); clones of any size
# result = infer_with_twinfer(cells, genes)                         # or regulators=[...], targets=[...]

# Network: scenarios 2-3 solid, scenario 4 dashed
plot_kwargs, edges = network_from_scenarios(result)
plot_grn(**plot_kwargs)
```

For count data, normalise before calling (for example log1p of counts per 10,000).

## Tutorials

[`tutorials/`](tutorials/) walks through the covariance decomposition, the inference scenarios, the choice of time point and number of twin pairs, direction inference, real-data input and network plots, on a simulation that ships with the repository.

## Repository layout

| Folder | Contents |
|---|---|
| [`package/`](package/) | the `twinfer` package: stochastic simulation of gene regulatory networks (`simulation`), the covariance decomposition and inference (`inference`), network plots (`plotting`) |
| [`tutorials/`](tutorials/) | worked examples |
| [`paper_analysis/`](paper_analysis/) | the code behind each figure of the paper (simulation, analysis and plot scripts), one folder per figure |
| `simulation_example_input_data/` | connectivity matrices and a parameter sheet for example simulations |
| `simulation_example_output_data/` | an example simulation (gene 1 → gene 2, 6,000 twin pairs) used by the tutorials |

## Reproducing the figures

Each folder in [`paper_analysis/`](paper_analysis/) has a README with the run order. Simulation outputs and analysis tables are too large for the repository; the scripts read and write them under a project folder set by `TWINFER_PROJECT_ROOT` (see [`paper_analysis/README.md`](paper_analysis/README.md)). The large simulations were run on a SLURM cluster; the launchers (`run_*.sh`, `run_*.sbatch`) are included.

## Citation

If you use TwINFER, please cite the preprint: bioRxiv [10.1101/2026.02.22.707230](https://doi.org/10.1101/2026.02.22.707230).

## License

MIT, see [LICENSE](LICENSE).
