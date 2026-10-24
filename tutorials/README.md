# Tutorials

| Notebook | What it covers |
|---|---|
| [`covariance_decomposition_and_inference_scenarios.ipynb`](covariance_decomposition_and_inference_scenarios.ipynb) | Scoring gene pairs with `infer_with_twinfer`: the three z-scores (`z_rho`, `z_rho_s`, `z_rho_delta`), the four inference scenarios, the effect of time after division and of the number of twin pairs, direction from two time points, real data with clones of any size, and drawing the result as a network (`network_from_scenarios` + `plot_grn`). |

The notebooks use the example simulation in `simulation_example_output_data/` (gene 1 activates gene 2, 6,000 twin pairs, 48 hours after division), so they run without downloading anything. They are saved with their outputs; run time is under a minute.

To run them, install the package first (see the [README](../README.md#installation)) and start Jupyter from the repository:

```bash
jupyter lab tutorials/
```
