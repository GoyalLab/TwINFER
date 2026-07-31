# heterogeneity_vs_regulation

Reproduces: "Twin difference correlations account for cell-state heterogeneity"
(Figure 2), and its associated extended-data panels.

## Question

Given a correlation between two genes across a cell population, determine
whether it arises from regulation, from cell-state heterogeneity, or both, using
a three-stage test on within-sample twin pairs at two time points (t1, t2).

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| simulate | `simulate.py` | Gene expression trajectories for the four scenarios (no regulation/single state, regulation/single state, regulation/multi-state, no regulation/multi-state) |
| analyze | `analyze.ipynb` | Stage I-III test statistics: gene-expression correlation, twin/random-pair difference correlation, Wilcoxon signed-rank test |
| plot | `plot.ipynb` | Flowchart panel, scenario boxplots, ROC curves for regulation and heterogeneity detection |

Also produced here:
- Scrambled-profile and random-pairing permutation-test distributions
- Brunner-Munzel test confirming genuinely distinct transcriptional states, used
  when filtering the parameter-space scan (see `paper_analysis/parameter_space_scan/`)
- Binomial vs. perfect partitioning comparison (`binomial_partition/` subfolder)

## Parameters

Gene expression: medians from Table E1. Regulatory interaction: defaults from
Table E2. Both tables are sourced in `paper_analysis/parameter_estimation/`.

| Constant | Value | Role |
|---|---|---|
| Stage I threshold | one-tailed p = 0.01 | Scrambled-profile permutation test |
| Stage II threshold | z-score = -10 | Random-pair permutation test |
| Stage III test | Wilcoxon signed-rank, p < 0.05 | Twin difference correlation, t1 vs t2 |
| n_cells | 6,000 twin pairs | Per simulation repeat |
| t1, t2 | 1h, 20h | Post-division sampling times |

## Depends on

`paper_analysis/parameter_space_scan/` for the ROC curves over 25,000
Latin-hypercube-sampled parameter sets.
