# TwINFER

TwINFER leverages information from recently divided sister cells, or *"twins"*, to distinguish between correlations arising from regulation and correlations from non-regulatory sources such as cell-state heterogeneity. Twin cells, immediately after division, belong to essentially the same state and the initial difference in the gene expression levels between them is minimal. Further, a cell approximates its twin cell's state and gene expression levels immediately after division. These two properties of twin cells enable TwINFER to *account for cell-state heterogeneity* and *infer causal relations*.

The same statistics work for lineage-tracing data with clones of any size, including singletons.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install the `twinfer` package.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Simulated twins, real lineage-tracing data and 10x input.
:::

:::{grid-item-card} API
:link: api
:link-type: doc

The functions used in the tutorials.
:::
::::

## How it works

For a pair of genes X and Y measured in N twin pairs, TwINFER splits the expression correlation into a part shared by twins and a part that differs between twins, and reports three analytic z-scores:

| z-score | measures |
|---|---|
| `z_rho` | the correlation of X and Y over all cells (Spearman) |
| `z_rho_s` | the twin-shared part: X in one sister against Y in the other sister |
| `z_rho_Delta` | the correlation of the twin differences of X and Y |

Each gene pair is then assigned an inference scenario:

| scenario | z_rho | z_rho_s | z_rho_Delta |
|---|---|---|---|
| 1 no regulation or cell heterogeneity identified | not significant | | |
| 2 regulation | significant | not significant | |
| 3 regulation + heterogeneity | significant | significant | significant |
| 4 heterogeneity | significant | significant | not significant |

The z-scores are tested in this order: `z_rho`, then `z_rho_s`, then `z_rho_Delta`. Significant means \|z\| > 1.96 for `z_rho` and `z_rho_Delta`, > 2.326 for `z_rho_s` (the thresholds can be changed). When many gene pairs are scored at once, as in real data, `z_rho` and `z_rho_Delta` are instead tested with a Benjamini-Hochberg correction: `z_rho` over all pairs, `z_rho_Delta` over the pairs whose `z_rho` is significant (q ≤ 0.05 for both).

When twins are measured at two time points, the cross-time correlation between twins (X in one sister earlier against Y in the other sister later, and the reverse) gives the direction of regulation.

## Citation

If you use TwINFER, please cite the preprint: bioRxiv [10.1101/2026.02.22.707230](https://doi.org/10.1101/2026.02.22.707230).

```{toctree}
:hidden:
:maxdepth: 2

installation
covariance_decomposition_derivation
tutorials/index
api
```
