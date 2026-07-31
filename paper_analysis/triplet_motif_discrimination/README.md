# triplet_motif_discrimination

Reproduces: "Twin cross-correlations distinguish between triplet motifs"
(Figure 4), and its associated extended-data panels.

## Question

Fan-out and feed-forward-loop motifs are frequently misclassified as a
regulated-mutual motif by correlation-based inference. Using the same
heterogeneity test from `heterogeneity_vs_regulation/`, determine whether an
inferred regulated-mutual edge is a false positive caused by an unobserved
confounding gene Z.

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| simulate | `simulate.py` | Fan-out, feed-forward-loop, and regulated-mutual triplets; a 14-gene network combining all three |
| analyze | `analyze.ipynb` | Twin cross-correlations vs. t2, z-scores from the difference-correlation permutation test, precision/recall/F1 vs. GRNBoost2 |
| plot | `plot.ipynb` | Motif schematics, cross-correlation curves, inferred network comparisons |

Also produced here:
- Sub-motif decomposition of the regulated-mutual motif (fan-in, feed-forward-loop,
  single/bidirectional edge), isolating fan-in as the source of correlation
  attenuation

## Parameters

| Constant | Value | Role |
|---|---|---|
| Z-score threshold | -12 | Separates fan-out/feed-forward-loop from regulated-mutual |
| t1 | 1h | Default measurement time |
| Network repeats | 10 | Precision/recall/F1 mean and SD |

## Depends on

`paper_analysis/grnboost2_comparison/` for the GRNBoost2 baseline and its
Gaussian-Mixture-Model threshold.
