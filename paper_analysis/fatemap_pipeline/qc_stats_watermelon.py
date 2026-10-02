#!/usr/bin/env python3
"""Per-sample n_genes / mito% distributions for Watermelon, and how many cells each proposed
threshold removes -- to check whether a 2500-gene floor is sensible."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
from paper_analysis.fatemap_pipeline.build_watermelon_qc_matrix import MAX_GENES, MITO_PCT, MIN_GENES, SAMPLES, load_sample
from paper_analysis.fatemap_pipeline.fatemap_qc_utils import apply_qc_filters

rows = []
for s in SAMPLES:
    a = load_sample(s)
    q = apply_qc_filters(a.X, a.var_names, MITO_PCT[s], MIN_GENES[s], MAX_GENES[s])
    g, m, t = q["n_genes"], q["pct_counts_mt"], q["total_counts"]
    rows.append({"sample": s, "cells": len(g),
                 "genes_q1": np.percentile(g, 1), "genes_q5": np.percentile(g, 5), "genes_q25": np.percentile(g, 25),
                 "genes_q50": np.median(g), "genes_q95": np.percentile(g, 95), "genes_q99": np.percentile(g, 99),
                 "frac_genes<1000": (g < 1000).mean(), "frac_genes<2000": (g < 2000).mean(),
                 "frac_genes<2500": (g < 2500).mean(), "frac_genes<3000": (g < 3000).mean(),
                 f"frac_genes>max": (g > MAX_GENES[s]).mean(),
                 "mito_q50": np.median(m), "mito_q95": np.percentile(m, 95), "frac_mito>cut": (m > MITO_PCT[s]).mean(),
                 "frac_pass_mito+genes": (q["pass_mito"] & q["pass_genes"]).mean(),
                 "umi_q50": np.median(t)})
df = pd.DataFrame(rows).set_index("sample")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(df.round(3).T.to_string(), flush=True)
df.round(4).to_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/watermelon/data/watermelon_rna_qc_stats.csv')
