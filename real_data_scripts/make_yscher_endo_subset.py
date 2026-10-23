#!/usr/bin/env python3
"""Subset of yscher's endo h5ad used by the PDF's Table 2: scored sample T15, cells with a clone_c5 barcode, 'spliced' layer.
Writes a cells x genes counts mtx, genes.txt and obs_metadata.csv (fatemap_clone_singletcode = clone_c5) for twinscore_pdf_candidates.py --qc-dir.
usage: make_yscher_endo_subset.py <out_dir> [h5ad]"""
import sys, numpy as np, pandas as pd, h5py, scipy.sparse as sp, scipy.io as sio
H = sys.argv[2] if len(sys.argv) > 2 else "/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/hPSC_endo_yscher_data/endo_kbnac_spliced_unspliced.h5ad"
out = sys.argv[1]
h = h5py.File(H, "r")
def ix(g): G = h[g]; k = G.attrs.get("_index", "_index"); return np.array(G[k]).astype(str)
def col(c): G = h["obs"][c]; cats = np.array(G["categories"]).astype(str); codes = np.array(G["codes"]); return np.where(codes < 0, "nan", cats[np.clip(codes, 0, len(cats) - 1)])
VN, ON, tp, cl = ix("var"), ix("obs"), col("timepoint"), col("clone_c5")
rr = np.where((tp == "T15") & (cl != "nan"))[0]
L = h["layers"]["spliced"]; ip = np.array(L["indptr"]); dat, ind = L["data"], L["indices"]
rows, cols, vals = [], [], []
for k, r in enumerate(rr):
    a, b = int(ip[r]), int(ip[r + 1]); rows.append(np.full(b - a, k)); cols.append(np.array(ind[a:b])); vals.append(np.array(dat[a:b], dtype=np.float32))
X = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(rr), len(VN))).tocsr()
sio.mmwrite(f"{out}/endo_T15_counts.mtx", X); open(f"{out}/genes.txt", "w").write("\n".join(VN) + "\n")
pd.DataFrame({"fatemap_clone_singletcode": cl[rr], "sample": "T15"}, index=[f"T15_{x}" for x in ON[rr]]).to_csv(f"{out}/obs_metadata.csv")
print("T15 cells with clone_c5:", len(rr), "| genes", len(VN))
