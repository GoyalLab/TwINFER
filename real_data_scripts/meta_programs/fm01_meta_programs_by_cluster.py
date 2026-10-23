#!/usr/bin/env python3
"""fm01_meta_programs.py run inside each annotated FM01 cluster (obs column 'cluster' of FM01_integrated.h5ad).
Unit = (cluster, replicate). Within a cluster the two replicates A and B play the role of the paper's two samples, so the
'robust across samples' test compares replicate A with replicate B of the same cluster. A unit with fewer than --min-cells
cells is not run (NMF at K up to 9 on a few dozen cells is not meaningful). Outputs per cluster under OUT/<cluster>/ plus
OUT/summary_by_cluster.csv.   usage: fm01_meta_programs_by_cluster.py [--min-cells 100]"""
import argparse, json, os, re
import numpy as np, pandas as pd, anndata as ad
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm01_meta_programs as M

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{M.ROOT}/analysis_data/{M.DS.lower()}/data/mp_nmf_by_cluster")
ap.add_argument("--min-cells", type=int, default=100)
a = ap.parse_args()
cl = ad.read_h5ad(f"{M.ROOT}/finalized_data/{M.DS}_data/{M.DS}_integrated.h5ad", backed="r").obs["cluster"].astype(str).str.strip()
X, genes, rep = M.load()
cells = np.array([l.strip() for l in open(f"{M.QC}/cells.txt")])
cl = cl.loc[cells].to_numpy()
ks = [4, 5, 6, 7, 8, 9]
summ = []
for c in sorted(set(cl)):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_")
    d = f"{a.out}/{slug}"; os.makedirs(d, exist_ok=True)
    n = {s: int(((cl == c) & (rep == s)).sum()) for s in ("A", "B")}
    row = dict(cluster=c, cells_A=n["A"], cells_B=n["B"])
    if min(n.values()) < a.min_cells:
        row.update(status=f"skipped (a replicate has < {a.min_cells} cells)"); summ.append(row); M.log(c, row["status"]); continue
    rows = []
    for s in ("A", "B"):
        sel = np.where((cl == c) & (rep == s))[0]
        Mx, gn = M.sample_matrix(X, genes, sel)
        M.log(c, s, "cells", Mx.shape[0], "genes", Mx.shape[1])
        M.run_nmf(Mx, gn, ks, s, rows)
    df = pd.DataFrame(rows, columns=["sample", "K", "idx", "rank", "gene", "score"]); df.to_csv(f"{d}/nmf_programs_all.csv.gz", index=False)
    P = M.to_programs(df); R = M.robust(P, ["A", "B"]); mps = M.make_mps(R)
    pd.DataFrame([dict(sample=p["sample"], K=p["K"], idx=p["idx"], best_overlap_other_replicate=p["best_other"], genes=" ".join(p["genes"])) for p in R]).to_csv(f"{d}/robust_programs.csv", index=False)
    pd.DataFrame([dict(MP=i + 1, n_programs=len(m["members"]), members=";".join(f"{s}_K{k}_{j}" for s, k, j in m["members"]), genes=" ".join(m["genes"])) for i, m in enumerate(mps)]).to_csv(f"{d}/meta_programs.csv", index=False)
    row.update(status="run", programs=len(P), within_ok=sum(p["in_ok"] for p in P), across_ok=sum(p["across_ok"] for p in P), robust_nonredundant=len(R), n_mps=len(mps))
    summ.append(row); M.log(json.dumps(row))
pd.DataFrame(summ).to_csv(f"{a.out}/summary_by_cluster.csv", index=False)
