#!/usr/bin/env python3
"""Write the expression file for the Julia PIDC (NetworkInference.jl): tab-separated, genes x cells, header = cell ids.
Uses the SAME cells the Python PIDC competitor used: twinfer_input_fm06_<gs>.csv, subsampled to 8,000 cells with
numpy.random.default_rng(0).choice(n, 8000, replace=False) (run_competitors_fatemap.py PIDC_NSUB / SEED). Zero-variance genes dropped.
usage: prep_julia_pidc_fm06.py <gene_set> [n_genes_test]   (n_genes_test: keep only the first n genes and 300 cells, for a smoke test)"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys
import numpy as np, pandas as pd
gs = sys.argv[1] if len(sys.argv) > 1 else "correlation_high"
test = int(sys.argv[2]) if len(sys.argv) > 2 else 0
D = f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data'
df = pd.read_csv(f"{D}/twinfer_input_fm06_{gs}.csv")
cols = [c for c in df.columns if c.endswith("_mRNA")]
X = df[cols].to_numpy(float)
if len(X) > 8000:
    X = X[np.random.default_rng(0).choice(len(X), 8000, replace=False)]
genes = [c[:-5] for c in cols]
if test:
    genes, X = genes[:test], X[:300, :test]
E = pd.DataFrame(X.T, index=genes, columns=[f"c{i}" for i in range(X.shape[0])])
const = E.index[E.nunique(axis=1) <= 1]
if len(const):
    print("dropping zero-variance genes:", list(const)); E = E.drop(index=const)
out = f"{D}/pidc_julia/{gs}" + ("_test" if test else "")
os.makedirs(out, exist_ok=True)
E.to_csv(f"{out}/ExpressionData.csv", sep="\t", header=True, index=True)
print(f"wrote {out}/ExpressionData.csv : {E.shape[0]} genes x {E.shape[1]} cells")
