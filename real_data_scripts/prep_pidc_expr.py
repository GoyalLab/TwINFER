#!/usr/bin/env python3
"""Expression table for the Julia PIDC (NetworkInference.jl): tab-separated genes x cells, cells named c0..cK.
usage: prep_pidc_expr.py <expr_genes_x_cells.csv> <out_dir> <N genes or 'all'> [n_cells=8000]
Genes: first N of a fixed random permutation (seed 0) of the expressed genes, so panels are nested. Cells: random sample (seed 0)."""
import sys, os
import numpy as np, pandas as pd
expr, out, N = sys.argv[1], sys.argv[2], sys.argv[3]
nc = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
genes = pd.read_csv(expr, usecols=[0]).iloc[:, 0].tolist()
ncell = len(pd.read_csv(expr, nrows=1).columns) - 1
order = np.random.default_rng(0).permutation(len(genes))
keep = set(order[: (len(genes) if N == "all" else int(N))].tolist())
cols = np.sort(np.random.default_rng(0).choice(ncell, min(nc, ncell), replace=False))
rows = []
for s in range(0, len(genes), 1000):
    ch = pd.read_csv(expr, index_col=0, skiprows=range(1, s + 1), nrows=1000)
    sel = [i - s for i in range(s, s + len(ch)) if i in keep]
    rows.append(ch.iloc[sel, cols])
E = pd.concat(rows)
E.columns = [f"c{i}" for i in range(E.shape[1])]
E = E.drop(index=E.index[E.nunique(axis=1) <= 1])
os.makedirs(out, exist_ok=True)
E.to_csv(f"{out}/ExpressionData.csv", sep="\t", header=True, index=True)
print(f"{E.shape[0]} genes x {E.shape[1]} cells -> {out}/ExpressionData.csv")
