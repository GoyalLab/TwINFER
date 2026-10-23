#!/usr/bin/env python3
"""Name the heritable identity axis (list_cell_state_axes.py) with the paper's own FateMap melanoma marker sets (resources/mmc4.xlsx, Table S4, dataset 'FateMap Melanoma': Melanocytic, Canonical Resistant,
Neural Crest, Smooth Muscle, Adhesive, type-1 interferon). Axis score of a cell = projection of its z-scored log2(CP10k/10+1) on the saved PC1 loadings of the 2,000 HVGs (<dataset>_identity_axis_genes.csv; the
depth/cycle residualisation used to find the axis is not redone here, so this is the same axis up to a small approximation). Marker-set score = mean z-scored expression of its genes.
Reports (1) correlation of every marker set with the axis across cells and across clone means (clones with >= 2 cells), (2) the axis loading of each marker gene, (3) which marker genes are in the top-100 unshared / shared lists.
usage: identity_axis_vs_melanoma_markers.py [FM01 FM06]"""
import sys
import numpy as np, pandas as pd, scipy.io as sio
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; OUT = f"{A}/fm_meta_programs_comparison/cell_state_axes"; L = f"{A}/fm_meta_programs_comparison/top100_lists"
T = pd.read_excel("/home/gzu5140/TwINFER_KA/resources/mmc4.xlsx", header=2); T.columns = ["dataset", "species", "tissue", "celltype", "genes", "ref"]; T = T[T.dataset == "FateMap Melanoma"]
SETS = {r.celltype.strip(): [g.strip() for g in str(r.genes).split(",") if g.strip() and g.strip() != "PKDVV"] for r in T.itertuples()}   # 'PKDVV' is not a gene symbol (typo in the table), dropped
z = lambda v: (v - v.mean()) / (v.std() if v.std() > 0 else 1)
for ds in (sys.argv[1:] or ["FM01", "FM06"]):
    qc = f"{ROOT}/finalized_data/{ds}_data/qc_filtered"; genes = np.array(open(f"{qc}/genes.txt").read().split()); cells = np.array(open(f"{qc}/cells.txt").read().split())
    X = sio.mmread(f"{qc}/{ds.lower()}_qc_counts.mtx").tocsr().astype(np.float32); obs = pd.read_csv(f"{qc}/obs_metadata.csv", index_col=0).loc[cells]
    cl = obs["fatemap_clone_singletcode"].astype(str).to_numpy(); ok = (cl != "nan") & (pd.Series(cl).str.len() > 0).to_numpy(); X = X[np.flatnonzero(ok)]; cl = cl[ok]
    nc = pd.Series(cl).map(pd.Series(cl).value_counts()).to_numpy(); k = nc >= 2; X = X[np.flatnonzero(k)]; cl = cl[k]
    tot = np.asarray(X.sum(1)).ravel(); cp = X.multiply(1e4 / tot[:, None]).tocsr(); cp.data = np.log2(cp.data / 10 + 1); gi = {g: i for i, g in enumerate(genes)}
    ld = pd.read_csv(f"{OUT}/{ds}_identity_axis_genes.csv").set_index("gene").loading; hv = [gi[g] for g in ld.index]
    Zh = (lambda M: (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1))(cp[:, hv].toarray()); idn = Zh @ ld.to_numpy()
    cid = pd.factorize(cl)[0]; cnt = np.bincount(cid); cmean = lambda v: np.bincount(cid, weights=v) / cnt
    U = {l: set(pd.read_csv(f"{L}/{ds}_{l}.csv").gene) | set(pd.read_csv(f"{L}/{ds}_{l}.csv").TF) for l in ("unshared", "shared")}
    print(f"\n=== {ds}: cells in clones >=2: {len(cl):,}; clones {len(cnt):,}")
    print(" marker set (genes) | corr with identity axis: cells / clone means | axis loading per gene (+ = same end as MLANA/PMEL if positive) | in top-100 lists")
    for name, gs in SETS.items():
        g_ok = [g for g in gs if g in gi]; s = np.mean([z(cp[:, gi[g]].toarray().ravel()) for g in g_ok], axis=0) if g_ok else None
        if s is None: print(f"  {name}: no genes in data"); continue
        r_cell = np.corrcoef(s, idn)[0, 1]; r_clone = np.corrcoef(cmean(s), cmean(idn))[0, 1]
        lds = ", ".join(f"{g} {ld[g]:+.3f}" if g in ld.index else f"{g} (not HVG)" for g in g_ok); lists = ", ".join(f"{g}:{'/'.join(l for l in U if g in U[l])}" for g in g_ok if any(g in U[l] for l in U)) or "none"
        print(f"  {name} ({','.join(g_ok)}) | {r_cell:+.2f} / {r_clone:+.2f} | {lds} | {lists}")
