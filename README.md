# TwINFER

TwINFER leverages information from recently divided sister cells, or “twins”, to infer gene regulatory networks: twin information discriminates regulatory from non-regulatory correlations and resolves interaction direction and type (activation/repression). Twin cells, immediately after division, belong to essentially the same state and the initial difference in the gene expression levels between them is minimal. Further, a cell approximates its
twin cell’s state and gene expression levels immediately after division. These two properties of twin cells enables TwINFER to *account for cell-state heterogeneity* and *infer causal relations*.

This repository contains the `twinfer` Python package, tutorials, and the code behind the figures of the paper.

## How it works

For a pair of genes X and Y measured in N twin pairs, TwINFER decomposes the correlation between genes into two components, shared and unshared between twins. It reports these metrics through three analytic z-scores:

| z-score | measures |
|---|---|
| `z_rho` | the correlation of X and Y over all cells (Spearman) |
| `z_rho_s` | the twin-shared part: X in one sister against Y in the other sister |
| `z_rho_Delta` | the correlation of the twin differences of X and Y |

For a more detailed description and derivation of these scores, please read [`covariance_decomposition_derivation.ipynb`](/docs/covariance_decomposition_derivation.ipynb).

Each gene pair is then assigned an inference scenario:

| scenario | $z_\rho$ | $z_s$ | $z_\Delta$ | reading |
|---|---|---|---|---|
| 1 nothing | not significant | | | no correlation |
| 2 regulation | significant | not significant | | correlation not shared by twins |
| 3 regulation + heterogeneity | significant | significant | significant | a shared part and a twin-difference part |
| 4 heterogeneity | significant | significant | not significant | correlation only through a shared state |

The z-scores are tested in this order: `z_rho`, then `z_rho_s`, then `z_rho_Delta`. Significant means \|z\| > 1.96 for `z_rho` and `z_rho_Delta`, > 2.326 for `z_rho_s` (the thresholds can be changed). When many gene pairs are scored at once, as in real data, `z_rho` and `z_rho_Delta` are instead tested with a Benjamini-Hochberg correction: `z_rho` over all pairs, `z_rho_Delta` over the pairs whose `z_rho` is significant (q ≤ 0.05 for both).

When twins are measured at two time points, the cross-time correlation between twins (X in one sister earlier against Y in the other sister later, and the reverse) gives the direction of regulation.

The same statistics can be extended to lineage-tracing data with clones of any size, including singletons by assigning equal weights to all clones.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/GoyalLab/TwINFER.git
cd TwINFER
conda create -n twinfer python=3.12
conda activate twinfer
pip install -e package/          # installs twinfer and its dependencies
pip install jupyterlab anndata   # only to run the tutorials
```

The example real dataset (`real_data/LSK_d2_d4_d6.h5ad`, 238 MB) is stored with [Git LFS](https://git-lfs.com). If it is only a small text file after cloning, run `git lfs install` once and then `git lfs pull` in the repository.

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
table = pd.read_csv("simulation_example_output_data/df_rows_0_0_24102026_140638_ncells_6000_A_to_B_04cf9c46.csv")
genes = ["gene_1_mRNA", "gene_2_mRNA"]

result = infer_with_twinfer(table, genes, time_points=[20])          # z_rho, z_rho_s, z_rho_Delta, inference_scenario per gene pair
result = infer_with_twinfer(table, genes, time_points=[1, 20])       # adds the direction (z_dag_xy, z_dag_yx, delta, p_delta)

# Real data: one row per cell with clone_id and normalised expression (no replicate column); clones of any size
# result = infer_with_twinfer(cells, genes)                         # or regulators=[...], targets=[...]

# Network: scenarios 2-3 solid, scenario 4 dashed
plot_kwargs, edges = network_from_scenarios(result)
plot_grn(**plot_kwargs)
```

For count data, normalise before calling (for example log1p of counts per 10,000).

## Tutorials

The [`tutorials/`](tutorials/) run with the data in this repository:

| Tutorial | What it covers | Data |
|---|---|---|
| [`inference_simulation_one_time_point.ipynb`](tutorials/inference_simulation_one_time_point.ipynb) | Twins measured at one time point: the three z-scores and the covariance decomposition, the inference scenarios, the effect of the number of twin pairs, network plots | example simulation (gene 1 → gene 2, 6,000 twin pairs) |
| [`inference_simulation_two_time_points.ipynb`](tutorials/inference_simulation_two_time_points.ipynb) | Twins measured at two time points (1 and 20 hours, as in the paper): cross-time correlation and the direction of regulation, which pairs to test, ranked lists, directed networks | the same simulation |
| [`inference_real_data_lineage_tracing.ipynb`](tutorials/inference_real_data_lineage_tracing.ipynb) | Real lineage-tracing data: from an AnnData object with clone labels to the input table, Benjamini-Hochberg-corrected scores, inference scenarios, ranked gene pairs, networks, several samples. The dataset-dependent choices are marked, with what is done for the example | LARRY (Weinreb et al., *Science* 2020), mouse haematopoiesis, days 2, 4, 6 (`real_data/`) |
| [`convert_10x_to_twinfer_input.ipynb`](tutorials/convert_10x_to_twinfer_input.ipynb) | From 10x output and a barcode / clone table to the input table (template) | your own 10x data |

## Repository layout

| Folder | Contents |
|---|---|
| [`package/`](package/) | the `twinfer` package: stochastic simulation of gene regulatory networks (`simulation`), the covariance decomposition and inference (`inference`), network plots (`plotting`) |
| [`tutorials/`](tutorials/) | worked examples |
| [`paper_analysis/`](paper_analysis/) | the code behind each figure of the paper (simulation, analysis and plot scripts), one folder per figure |
| `simulation_example_input_data/` | connectivity matrices and a parameter sheet for example simulations |
| `simulation_example_output_data/` | an example simulation (gene 1 → gene 2, 6,000 twin pairs) used by the simulation tutorials |
| `real_data/` | an example lineage-tracing dataset: LARRY, mouse haematopoietic progenitors on days 2, 4 and 6 (Weinreb et al., *Science* 2020), `LSK_d2_d4_d6.h5ad`, stored with Git LFS; used by the real-data tutorial |

## Citation

If you use TwINFER, please cite the preprint: bioRxiv [10.64898/2026.02.22.707230](https://doi.org/10.64898/2026.02.22.707230).

## License

MIT, see [LICENSE](LICENSE).
