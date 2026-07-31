# grnboost2_comparison

Reproduces: "Thresholding the GRNBoost2 ranked list using a Gaussian Mixture
Model" (Methods, extended-data panel).

Used by: `causal_direction_inference/` (Figure 3d-e), `triplet_motif_discrimination/`
(Figure 4e-f).

## Question

GRNBoost2 produces a ranked, unthresholded list of candidate interactions.
This provides a reproducible thresholding method (two-component Gaussian
Mixture Model on the importance-score distribution) plus an upper-bound
threshold (best possible F1, computed retrospectively) for comparison.

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| run | `run_grnboost2.py` | GRNBoost2 importance scores via Arboreto's `grnboost2` function |
| threshold | `threshold.py` | GMM-based threshold, retrospective best-F1 upper-bound threshold |

## Environment

Requires the `grnboost_env` conda environment (`package/grnboost_env.yml`),
not `twinfer-code` — GRNBoost2/Arboreto depend on older numpy/pandas/scipy
pins.

## Parameters

| Constant | Value | Role |
|---|---|---|
| GMM components | 2 | Separates low- and high-importance interactions |
| Threshold rule | Intersection of the two weighted Gaussian densities | Edge equally likely to belong to either component |
