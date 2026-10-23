#!/usr/bin/env python3
"""genes x cells log1p(CP10k) table from a subset folder (endo_T15_counts.mtx cells x genes, genes.txt, obs_metadata.csv); genes detected in >= 5% of the cells.
usage: prep_expr_from_subset.py <subset_dir> <mtx name> <out_csv> [min_det=0.05]"""
import sys, numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp
d, mtx, out = sys.argv[1], sys.argv[2], sys.argv[3]; md = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
X = sio.mmread(f"{d}/{mtx}").tocsr().astype(np.float32); g = np.array(open(f"{d}/genes.txt").read().split()); obs = pd.read_csv(f"{d}/obs_metadata.csv", index_col=0)
tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
X = sp.diags(1e4 / tot) @ X; X.data = np.log1p(X.data); X = X.tocsc()
det = np.asarray((X > 0).mean(0)).ravel(); keep = np.flatnonzero(det >= md); keep = keep[~pd.Index(g[keep]).duplicated()]
print("cells", X.shape[0], "genes kept", len(keep), "of", len(g))
with open(out, "w") as fh:
    fh.write("," + ",".join(obs.index) + "\n")
    for s in range(0, len(keep), 500):
        idx = keep[s:s + 500]; pd.DataFrame(X[:, idx].toarray().T, index=g[idx]).round(4).to_csv(fh, header=False)
