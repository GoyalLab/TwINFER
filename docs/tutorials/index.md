# Tutorials

Worked examples of TwINFER, rendered from the notebooks in [`tutorials/`](https://github.com/GoyalLab/TwINFER/tree/TwINFER_v2/tutorials). They run with the data in the repository; see [Installation](../installation.md) to set them up.

## Simulation

Simulate a population of twin cells with a known gene regulatory network, using {mod}`twinfer.simulation`.

- [Simulating twin data](simulation_twin_data.ipynb): the connectivity matrix, the parameter file, the configuration dictionary, running the Gillespie simulation, the output, and confirming steady state.

## Inference on simulated twins

Score gene pairs of the example simulation (gene 1 activates gene 2, 6,000 twin pairs) with {func}`~twinfer.inference.covariance_decomposition_z_scores.infer_with_twinfer`.

- [One time point](inference_simulation_one_time_point.ipynb): the three z-scores and the covariance decomposition, the inference scenarios, the effect of the number of twin pairs, network plots.
- [Two time points](inference_simulation_two_time_points.ipynb): twins measured at 1 and 20 hours, cross-time correlation and the direction of regulation, which pairs to test, ranked lists, directed networks.

## Real data

Score lineage-tracing data with clones of any size.

- [Lineage-traced single cells](inference_real_data_lineage_tracing.ipynb): LARRY, mouse haematopoiesis, days 2, 4, 6. From an AnnData object with clone labels to the input table, Benjamini-Hochberg-corrected scores, inference scenarios, ranked gene pairs, networks, several samples.
- [From 10x output to TwINFER input](convert_10x_to_twinfer_input.ipynb): a template from 10x output and a barcode / clone table to the input table, with normalisation.

```{toctree}
:hidden:
:caption: Simulation

simulation_twin_data
```

```{toctree}
:hidden:
:caption: Inference on simulated twins

inference_simulation_one_time_point
inference_simulation_two_time_points
```

```{toctree}
:hidden:
:caption: Real data

inference_real_data_lineage_tracing
convert_10x_to_twinfer_input
```
