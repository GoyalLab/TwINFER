#!/usr/bin/env python3
"""Expression tables for the Julia-PIDC scaling test (FM06, log1p CP10k of the QC counts, 8,000 randomly chosen QC cells, seed 0).
Genes = the CollecTRI source genes detected in > 5% of QC cells (581; list from fm06_gene_detection_pct.csv), taken in a fixed random
order (seed 0); the size-N table uses the first N of that order (nested panels).
usage: prep_julia_pidc_scaling.py N [N ...]   -> <BASE>/analysis_data/fm06/data/pidc_julia/scaling_N<N>/ExpressionData.csv (tab-separated, genes x cells)"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys
import numpy as np, pandas as pd, scipy.io as sio
B = f'{TWINFER_PROJECT_ROOT}'
Q = f"{B}/finalized_data/FM06_data/qc_filtered"
det = pd.read_csv(f"{B}/analysis_data/fatemap_comparison/data/fm06_gene_detection_pct.csv")
pool = det[(det.collectriSrc) & (det.pct_all > 5)].gene.tolist()
order = list(np.random.default_rng(0).permutation(pool))
sizes = [int(a) for a in sys.argv[1:]] or [581]
assert max(sizes) <= len(order), (max(sizes), len(order))
X = sio.mmread(f"{Q}/fm06_qc_counts.mtx").tocsr()
genes_all = np.array(open(f"{Q}/genes.txt").read().split())
gidx = {g: i for i, g in enumerate(genes_all)}
tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
cells = np.random.default_rng(0).choice(X.shape[0], 8000, replace=False)
print(f"pool {len(pool)} genes; {X.shape[0]} QC cells -> 8000 sampled")
for n in sizes:
    gl = order[:n]
    M = X[cells][:, [gidx[g] for g in gl]].toarray().astype(float)
    M = np.log1p(M / np.where(tot[cells] == 0, 1, tot[cells])[:, None] * 1e4)
    E = pd.DataFrame(M.T, index=gl, columns=[f"c{i}" for i in range(M.shape[0])])
    const = E.index[E.nunique(axis=1) <= 1]
    if len(const): E = E.drop(index=const)
    out = f"{B}/analysis_data/fm06/data/pidc_julia/scaling_N{n}"
    os.makedirs(out, exist_ok=True)
    E.to_csv(f"{out}/ExpressionData.csv", sep="\t", header=True, index=True)
    print(f"N={n}: wrote {E.shape[0]} genes x {E.shape[1]} cells -> {out}", flush=True)
