# larry_hematopoiesis_validation

Reproduces: "TwINFER captures regulatory interactions and confounding
heterogeneity in experimental data" (Figure 5), and its associated
extended-data panels.

## Question

Apply TwINFER to a real, lineage-barcoded scRNA-seq dataset (LSK LARRY,
Weinreb et al. 2020) to test whether the framework recovers known
regulatory interactions and flags heterogeneity in an experimental setting,
using clonal pairs as a proxy for twin pairs.

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| preprocess | `preprocess.py` | Filtered, normalized LARRY dataset; doublet removal via singletCode; clonal pairing by barcode |
| analyze | `analyze.ipynb` | Differential expression, transcriptomic-distance comparison (clonal vs. random pairs), heterogeneity/regulation classification per gene pair, stable-correlation filtering across time points |
| plot | `plot.ipynb` | Volcano plot, dot matrices, co-expression vs. cross-correlation matrices |

## Data

LSK LARRY dataset (Weinreb et al. 2020), days 2/4/6 post-barcoding. Not
redistributed in this repository; see `paper_analysis/README.md` for the
source location.

## Filters applied

- Cells with >20% mitochondrial gene content removed
- Lineage barcodes merged within Hamming distance 3
- Doublets removed via singletCode
- Cells expressing <200 genes discarded
- Gene pairs excluded if either: (a) not part of the stable-correlation subnetwork
  (significant and same-sign across all three days), or (b) shuffle-test
  distribution fails a normality check (Q-Q R-squared < 0.9)

## Depends on

None. Validation is against literature-reported interactions (Cebpa->Spi1,
Spi1->Jun), not a GRNBoost2/BEELINE comparison.
