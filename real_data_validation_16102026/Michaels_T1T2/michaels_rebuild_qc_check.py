#!/usr/bin/env python3
"""Rebuild yscher's michaels_T1_qc.h5ad / michaels_T2_qc.h5ad from the raw GEO matrices with an independent implementation of her
mich_qc.py + helpers/qc.py rule (mito <= 20% with 'MT-' genes, >= 200 genes) and compare with her files element by element.
Checks: obs_names (and order), var_names (and order), X (shape, dtype, indptr, indices, data - exact), obs columns, uns/obs extras.
Memory-light (streams the mtx); run on a login node."""
import json, sys
import h5py, numpy as np, pandas as pd, scipy.sparse as sp
from anndata.utils import make_index_unique
Y = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/datasets/michaels2024"
res = {}


def load_raw(g):
    bc = pd.read_csv(f"{Y}/raw/{g}_barcodes.tsv.gz", header=None)[0].to_numpy()
    ft = pd.read_csv(f"{Y}/raw/{g}_features.tsv.gz", header=None, sep="\t")
    ng, nc = len(ft), len(bc)
    cols, rows, vals = [], [], []
    last = 0
    for ch in pd.read_csv(f"{Y}/raw/{g}_matrix.mtx.gz", sep=" ", header=None, skiprows=3, chunksize=5_000_000, dtype=np.int64):
        c = ch[1].to_numpy() - 1
        assert c[0] >= last and (np.diff(c) >= 0).all(), "mtx not sorted by cell"
        last = c[-1]
        cols.append(c.astype(np.int32)); rows.append((ch[0].to_numpy() - 1).astype(np.int32)); vals.append(ch[2].to_numpy().astype(np.float32))
    c = np.concatenate(cols); del cols
    cnt = np.bincount(c, minlength=nc); indptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    idx = np.concatenate(rows); dat = np.concatenate(vals); del rows, vals, c
    X = sp.csr_matrix((dat, idx, indptr), shape=(nc, ng))
    X.sort_indices()
    return X, bc, ft


def h5_strs(d):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in d[()]])


for tag, g in (("T1", "GSM7916614_T1"), ("T2", "GSM7916615_T2")):
    X, bc, ft = load_raw(g)
    sym = ft[1].astype(str).to_numpy()
    var = make_index_unique(pd.Index(sym))
    # QC (independent of her helper): mito = genes starting with 'mt-' (any case), counts from X
    mt = np.char.startswith(np.char.lower(sym.astype(str)), "mt-")
    total = np.asarray(X.sum(1)).ravel().astype(float)
    ngen = np.asarray((X > 0).sum(1)).ravel().astype(np.int64)
    mito = np.asarray(X[:, mt].sum(1)).ravel().astype(float)
    pct = np.divide(100.0 * mito, total, out=np.zeros_like(total), where=total > 0)
    keep = ~((ngen < 200) | (pct > 20.0))
    Xk = X[np.flatnonzero(keep)].tocsr(); Xk.sort_indices()
    names = np.array([f"{tag}|{b}" for b in bc])[keep]
    r = dict(raw_cells=len(bc), kept=int(keep.sum()), n_mito_genes=int(mt.sum()))
    with h5py.File(f"{Y}/processed/michaels_{tag}_qc.h5ad", "r") as f:
        hx = f["X"]
        r["her_X_encoding"] = dict(hx.attrs).get("encoding-type", "?")
        hi, hp, hd = hx["indices"][()], hx["indptr"][()], hx["data"][()]
        r["her_shape"] = tuple(int(i) for i in hx.attrs["shape"]) if "shape" in hx.attrs else None
        hobs = f["obs"]; hvar = f["var"]
        her_names = h5_strs(hobs[hobs.attrs["_index"]]); her_var = h5_strs(hvar[hvar.attrs["_index"]])
        r["her_obs_columns"] = [k for k in hobs.keys() if k != hobs.attrs["_index"]]
        r["her_var_columns"] = [k for k in hvar.keys() if k != hvar.attrs["_index"]]
        r["her_uns_keys"] = list(f["uns"].keys()) if "uns" in f else []
        hcols = {k: hobs[k][()] for k in r["her_obs_columns"] if isinstance(hobs[k], h5py.Dataset)}
    r["obs_names_identical_incl_order"] = bool(len(her_names) == len(names) and (her_names == names).all())
    r["obs_names_same_set"] = bool(set(her_names) == set(names))
    r["var_names_identical_incl_order"] = bool(len(her_var) == len(var) and (her_var == np.asarray(var)).all())
    r["X_shape_identical"] = (Xk.shape == (len(her_names), len(her_var)))
    r["X_dtype_her"] = str(hd.dtype); r["X_dtype_mine"] = str(Xk.dtype)
    r["X_nnz_identical"] = bool(len(hd) == Xk.nnz)
    if r["obs_names_identical_incl_order"] and r["var_names_identical_incl_order"]:
        r["X_indptr_identical"] = bool(np.array_equal(hp, Xk.indptr))
        r["X_indices_identical"] = bool(np.array_equal(hi, Xk.indices))
        r["X_data_identical_exact"] = bool(np.array_equal(hd, Xk.data))
        r["X_max_abs_diff"] = float(np.abs(hd.astype(np.float64) - Xk.data.astype(np.float64)).max()) if len(hd) == Xk.nnz else None
        # obs columns she stored (n_genes_by_counts, pct_counts_mt, total_counts, timepoint ...)
        mine = {"n_genes_by_counts": ngen[keep], "pct_counts_mt": pct[keep], "total_counts": total[keep]}
        for k, v in mine.items():
            if k in hcols:
                r[f"obs_{k}_max_abs_diff"] = float(np.abs(hcols[k].astype(float) - v).max())
        r["her_timepoint_values"] = sorted(set(map(str, h5_strs(hobs["timepoint"]["categories"])))) if "timepoint" in hobs and isinstance(hobs["timepoint"], h5py.Group) else "n/a"
    res[tag] = r
    print(tag, json.dumps(r, indent=1, default=str), flush=True)
json.dump(res, open("/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/michaels_rerun/qc_h5ad_identity.json", "w"), indent=1, default=str)
