# Tutorials

| Notebook | What it covers | Data |
|---|---|---|
| [`inference_simulation_one_time_point.ipynb`](inference_simulation_one_time_point.ipynb) | Simulated twin pairs at one time point: the three z-scores (`z_rho`, `z_rho_s`, `z_rho_Delta`) and their decomposition, the four inference scenarios, the effect of the number of twin pairs, and network plots. | `simulation_example_output_data/` (gene 1 activates gene 2, 6,000 twin pairs) |
| [`inference_simulation_two_time_points.ipynb`](inference_simulation_two_time_points.ipynb) | Simulated twin pairs at two time points (1 and 20 hours, as in the paper): the three groups of clones, cross-time correlation and direction of regulation, other time points, which pairs to test (`cross_correlation_pairs`), the ranked cross-time list, and a directed network. | the same simulation |
| [`inference_real_data_lineage_tracing.ipynb`](inference_real_data_lineage_tracing.ipynb) | Real lineage-tracing data: from an AnnData object with clone labels to the input table (normalisation, cells, genes), scoring all gene pairs with Benjamini-Hochberg correction, inference scenarios, ranked lists, network, and comparing samples. The dataset-dependent choices are marked, with what is done for the example. | `real_data/LSK_d2_d4_d6.h5ad` (LARRY, mouse haematopoiesis, days 2, 4, 6; Git LFS) |
| [`convert_10x_to_twinfer_input.ipynb`](convert_10x_to_twinfer_input.ipynb) | From 10x output (`matrix.mtx`, `features.tsv`, `barcodes.tsv`) and a barcode / clone table to the input table, with normalisation; a template with placeholder paths. | your own 10x data |

Each tutorial saves everything it computes (score tables with their settings, ranked lists, network edges and figures) in `tutorials/tutorial_output/<tutorial>/`; this folder is not part of the repository. The first three notebooks are saved with their outputs and run with the data in this repository (under a minute each). The large data file is stored with Git LFS: after cloning, run `git lfs pull` if `real_data/LSK_d2_d4_d6.h5ad` is only a small text pointer.

To run them, install the package first (see the [README](../README.md#installation)) and start Jupyter from the repository:

```bash
jupyter lab tutorials/
```
