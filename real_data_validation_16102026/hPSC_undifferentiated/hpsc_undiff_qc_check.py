#!/usr/bin/env python3
"""hPSC_undifferentiated: recompute the QC from the Cell Ranger filtered matrices (mito < 20%, >= 200 genes) and compare with the cells and counts of our file.
Sources (from yscher's reproduction notes): T0 cellranger_out/T0h, T10 and T15 cellranger_10x/{T10h,T15h}. Streams the 10x h5 in cell chunks (small memory).
Compared with: real_data/twinfer2026_run1plus2_counts_mito20_min200genes.h5ad (integer counts; same cells as finalized_data/hPSC_undifferentiated)."""
import json

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

R = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
B = "/projects/b1042/GoyalLab/Hanxiao/TwINFER_2026scRNAseq"
SRC = {"T0": f"{B}/cellranger_out/T0h", "T10": f"{B}/cellranger_10x/T10h", "T15": f"{B}/cellranger_10x/T15h"}
OUT = f"{R}/analysis_data/dataset_qc_verification/hpsc_undifferentiated_qc_check.json"
fo = h5py.File(f"{R}/real_data/twinfer2026_run1plus2_counts_mito20_min200genes.h5ad", "r")
obs, var = ad.io.read_elem(fo["obs"]), ad.io.read_elem(fo["var"])
Xg = fo["X"]
ip_o = Xg["indptr"][:]
gid_o = var["gene_ids"].astype(str).to_numpy()
rng = np.random.default_rng(0)
res = {}
for tp, d in SRC.items():
    f = h5py.File(f"{d}/outs/filtered_feature_bc_matrix.h5", "r"); m = f["matrix"]
    bc = np.array([x.decode() for x in m["barcodes"][:]]); gid = np.array([x.decode() for x in m["features/id"][:]]); nm = np.array([x.decode() for x in m["features/name"][:]])
    mito = np.array([g.upper().startswith("MT-") for g in nm]); ip = m["indptr"][:]; ng = len(gid); nc = len(bc)
    pm = np.zeros(nc); gn = np.zeros(nc, int); tot = np.zeros(nc)
    for a in range(0, nc, 3000):
        b = min(a + 3000, nc); s, e = ip[a], ip[b]
        C = sp.csc_matrix((m["data"][s:e], m["indices"][s:e], ip[a:b + 1] - s), shape=(ng, b - a))
        t = np.asarray(C.sum(0)).ravel().astype(float); mt = np.asarray(C[mito].sum(0)).ravel().astype(float)
        tot[a:b] = t; pm[a:b] = np.divide(100 * mt, t, out=np.zeros_like(t), where=t > 0); gn[a:b] = np.diff(C.indptr)
    keep = (pm < 20) & (gn >= 200)
    ours = obs[obs["timepoint"].astype(str) == tp]
    ours_ids = set(ours["cell_id"].astype(str))
    mine = set(bc[keep].tolist()) if False else set(x.replace("-1", "") for x in bc[keep])
    ours_bc = set(x.replace("-1", "") for x in ours_ids)
    # counts: 300 random kept cells, full gene vector compared with our file (aligned on gene id)
    pos = {x.replace("-1", ""): i for i, x in enumerate(bc)}
    orow = {c: i for i, c in zip(np.flatnonzero((obs["timepoint"].astype(str) == tp).to_numpy()), ours["cell_id"].astype(str))}
    common = sorted(set(pos) & set(x.replace("-1", "") for x in orow))
    samp = rng.choice(len(common), size=min(300, len(common)), replace=False); same = 0
    g2i = {g: i for i, g in enumerate(gid)}
    idx_o2c = np.array([g2i.get(g, -1) for g in gid_o])
    for k in samp:
        c = common[k]; ci = pos[c]; oi = [i for cc, i in orow.items() if cc.replace("-1", "") == c][0] if False else None
    # build lookup once
    orow2 = {k.replace("-1", ""): v for k, v in orow.items()}
    for k in samp:
        c = common[k]; ci = pos[c]
        s, e = ip[ci], ip[ci + 1]; v = np.zeros(ng); v[m["indices"][s:e]] = m["data"][s:e]
        oi = orow2[c]; s2, e2 = ip_o[oi], ip_o[oi + 1]; w = np.zeros(ng)
        ind = Xg["indices"][s2:e2]; dat = Xg["data"][s2:e2]; tgt = idx_o2c[ind]; ok = tgt >= 0; w[tgt[ok]] = dat[ok]
        same += int(np.array_equal(v, w))
    res[tp] = dict(source=d, cellranger_cells=nc, pass_mito20_min200genes=int(keep.sum()), our_cells=len(ours_bc), recomputed_set_equals_ours=bool(mine == ours_bc),
                   only_recomputed=len(mine - ours_bc), only_ours=len(ours_bc - mine), sampled_cells_compared=int(len(samp)), sampled_cells_identical_count_vector=same,
                   our_max_pct_mt=float(ours["pct_counts_mt"].max()), recomputed_max_pct_mt_of_kept=float(pm[keep].max()), min_genes_kept=int(gn[keep].min()))
    print(tp, json.dumps(res[tp]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
