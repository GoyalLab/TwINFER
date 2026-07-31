# parameter_space_scan

Reproduces: "Scanning the parameter space to define inference capabilities and
limitations" (Methods).

Used by: `heterogeneity_vs_regulation/` (Figure 2i-k), `causal_direction_inference/`
(Figure 3c).

## Question

Median/default parameters (Tables E1-E2) give TwINFER a perfect F1 score. This
scan tests performance across the full biologically-relevant parameter range
instead of only the defaults.

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| simulate | `simulate.py` | 25,000 Latin-hypercube-sampled parameter sets per scenario, ranges from Tables E1-E2 |
| analyze | `analyze.py` | ROC curves for regulation detection, multi-state detection, and heterogeneity detection |

## Parameters

| Constant | Value | Role |
|---|---|---|
| Sample size | 25,000 sets per scenario | Latin hypercube sampling |
| Heterogeneity criterion | Brunner-Munzel test, both genes p < 0.01 | Excludes parameter draws where two "states" are not statistically distinct (see `heterogeneity_vs_regulation/`) |

## Output

`<data_root>/paper_analysis/parameter_space_scan/<stage>/<run_tag>/`, consumed
by the two figure folders listed above via `twinfer.utils.paths.stage_dir()`.
