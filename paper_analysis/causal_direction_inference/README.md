# causal_direction_inference

Reproduces: "Twin cross-correlations allow inference of causal relations"
(Figure 3), and its associated extended-data panels.

## Question

Given a regulatory correlation between two genes, determine the direction
(X->Y vs Y->X) and type (activation vs repression) using cross-sample twin
correlations between x(t1) and y(t2), and y(t1) and x(t2).

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| simulate | `simulate.py` | All combinations of direction and type for a gene pair; a 5-gene linear cascade |
| analyze | `analyze.ipynb` | Cross-correlation matrices, permutation-test thresholds, precision/recall/F1 vs. GRNBoost2 |
| plot | `plot.ipynb` | Cross-correlation matrices, ROC curves, cascade inference comparison |

Also produced here:
- Scan over all (t1, t2) measurement-time pairs, for robustness
- Permutation test for the significance of a single directional interaction

## Parameters

| Constant | Value | Role |
|---|---|---|
| Threshold p-value | two-tailed p = 0.02 | Cross-correlation permutation test |
| t1, t2 | 1h, 20h | Default measurement times |
| n_cells | 6,000 twin pairs | Per simulation repeat |
| Cascade repeats | 10 | Precision/recall/F1 mean and SD |

## Depends on

- `paper_analysis/parameter_space_scan/` for the direction-inference ROC curves
  over 25,000 Latin-hypercube-sampled parameter sets.
- `paper_analysis/grnboost2_comparison/` for the GRNBoost2 baseline and its
  Gaussian-Mixture-Model threshold.
