#!/usr/bin/env python3
"""Meta-programs (MPs) of Gavish et al. 2023 (Nature 618:598, doi 10.1038/s41586-023-06130-4), run on FM01.
Orthogonal to TwinScore: no clone information is used.

Steps, following the Methods text the user pasted (2026-10-18):
  1. NMF per sample, K = 4..9 (39 programs per sample); each program = top 50 genes by NMF coefficient.
     Input per sample: expression centred per gene, negative values set to zero.
  2. Robust NMF programs, three criteria
       (a) robust within the sample: >= 35 of 50 genes shared with a program of the same sample at a different K
       (b) robust across samples:   >= 10 of 50 genes (20 %) shared with any program of any other sample
       (c) non-redundant within the sample: programs ranked by their best overlap with other samples' programs, taken in
           decreasing order; once taken, any other program of the sample sharing >= 10 genes (20 %) is dropped.
  3. MPs: clustering of the robust programs by gene overlap (founder = program with most overlaps of >= 10 genes, needs
     > 5 such overlaps; the program with the highest overlap with the founder starts the cluster; MP = genes in both,
     filled to 50 with the top NMF scores; then the program with the maximal overlap with the MP is added while the overlap
     is >= 10 genes, and the MP is updated to the genes present in most programs of the cluster, filled to 50 by summed
     NMF score; cluster closed when nothing can be added).
  The paper's further MP filters (ribosomal / mitochondrial enrichment, single-study clusters, doublet-like profiles) need
  more than one study and are reported as flags only.

Choices NOT fixed by the pasted text (set here, changeable by arguments):
  - sample = FM01 replicate (A, B); the paper's "tumours" are samples, so with FM01 only two samples exist
  - expression = log2(CP10k/10 + 1), genes detected in >= 5 % of the sample's cells (the TwinScore floor)
  - NMF: scikit-learn, init nndsvda, solver cd, max_iter 400, random_state 0
  - "several similar programs" in (a) is implemented as at least one other K
usage: fm01_meta_programs.py [--out DIR] [--max-cells N] [--ks 4,5,6,7,8,9] [--smoke]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from sklearn.decomposition import NMF

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
# (FM01 only; replaced by the dataset-aware lines below)
# QC = f"{ROOT}/finalized_data/FM01_data/qc_filtered"
DS = os.environ.get("FM_DATASET", "FM01")   # FM01, FM03, FM05, FM06, FM07, FM08
QC = f"{ROOT}/finalized_data/{DS}_data/qc_filtered"
TOP = 50
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def load():
    genes = np.array([l.strip() for l in open(f"{QC}/genes.txt")])
    cells = np.array([l.strip() for l in open(f"{QC}/cells.txt")])
    X = sio.mmread(f"{QC}/{DS.lower()}_qc_counts.mtx").tocsr().astype(np.float32)   # cells x genes
    assert X.shape == (len(cells), len(genes))
    obs = pd.read_csv(f"{QC}/obs_metadata.csv", index_col=0).loc[cells]
    return X, genes, obs.replicate.to_numpy()


def sample_matrix(X, genes, sel, det=0.05):
    S = X[sel]
    cp = S.multiply(1e4 / np.asarray(S.sum(1)).ravel()[:, None]).tocsr()
    cp.data = np.log2(cp.data / 10 + 1)
    keep = np.asarray((S > 0).mean(0)).ravel() >= det
    M = cp[:, keep].toarray()
    M -= M.mean(0, keepdims=True)
    M[M < 0] = 0
    return M.astype(np.float32), genes[keep]


def run_nmf(M, gnames, ks, tag, rows):
    for k in ks:
        t = time.time()
        m = NMF(n_components=k, init="nndsvda", solver="cd", max_iter=400, random_state=0)
        m.fit(M)
        for i, h in enumerate(m.components_):
            top = np.argsort(-h)[:TOP]
            for r, j in enumerate(top):
                rows.append((tag, k, i, r, gnames[j], float(h[j])))
        log(f"{tag} K={k} done {time.time()-t:.0f}s")


def to_programs(df):
    P = []
    for (s, k, i), g in df.groupby(["sample", "K", "idx"], sort=True):
        g = g.sort_values("rank")
        P.append(dict(sample=s, K=int(k), idx=int(i), genes=list(g.gene), score=dict(zip(g.gene, g.score))))
    return P


# def ov(a, b):   # replaced: failed when exactly one argument was a program dict and the other a gene list (reached only once a cluster forms)
#     return len(set(a["genes"]) & set(b["genes"])) if isinstance(a, dict) else len(set(a) & set(b))
def ov(a, b):
    ga = a["genes"] if isinstance(a, dict) else a
    gb = b["genes"] if isinstance(b, dict) else b
    return len(set(ga) & set(gb))


def robust(P, samples):
    for p in P:
        p["in_ok"] = any(q is not p and q["sample"] == p["sample"] and q["K"] != p["K"] and ov(p, q) >= 35 for q in P)
        o = [ov(p, q) for q in P if q["sample"] != p["sample"]]
        p["best_other"] = max(o) if o else 0
        if len(samples) == 1:   # one sample (pooled mode, FM03): rank by the best overlap with a program of the same sample at another K
            p["best_other"] = max([ov(p, q) for q in P if q is not p and q["K"] != p["K"]] or [0])
        p["across_ok"] = p["best_other"] >= 10 if len(samples) > 1 else True   # one sample (FM03): criterion (b) cannot be tested
    keep = []
    for s in samples:
        cand = sorted([p for p in P if p["sample"] == s and p["in_ok"] and p["across_ok"]], key=lambda p: -p["best_other"])
        chosen = []
        for p in cand:
            if all(ov(p, c) < 10 for c in chosen):
                chosen.append(p)
        keep += chosen
    return keep


def make_mps(R):
    R = list(R)
    mps = []
    while len(R) > 1:
        cnt = [sum(1 for q in R if q is not p and ov(p, q) >= 10) for p in R]
        j = int(np.argmax(cnt))
        if cnt[j] <= 5:
            break
        f = R[j]
        others = [q for q in R if q is not f]
        q0 = max(others, key=lambda q: ov(f, q))
        cl = [f, q0]

        def build(cl):
            c = {}
            for p in cl:
                for g in p["genes"]:
                    c[g] = c.get(g, 0) + 1
            sc = {}
            for p in cl:
                for g, v in p["score"].items():
                    sc[g] = sc.get(g, 0) + v
            return sorted(c, key=lambda g: (-c[g], -sc[g]))[:TOP]

        mp = build(cl)
        while True:
            rest = [q for q in R if q not in cl]
            if not rest:
                break
            q = max(rest, key=lambda q: ov(q, mp))
            if ov(q, mp) < 10:
                break
            cl.append(q)
            mp = build(cl)
        mps.append(dict(genes=mp, members=[(p["sample"], p["K"], p["idx"]) for p in cl]))
        R = [q for q in R if q not in cl]
    return mps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/analysis_data/{DS.lower()}/data/mp_nmf")
    ap.add_argument("--max-cells", type=int, default=0, help="subsample cells per sample (smoke test only)")
    ap.add_argument("--ks", default="4,5,6,7,8,9")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pool", action="store_true", help="treat ALL cells as one sample (no across-sample test; ranks programs by within-sample stability)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ks = [int(k) for k in a.ks.split(",")]
    X, genes, rep = load()
    if a.pool:
        rep = np.array(["ALL"] * len(rep))
    samples = sorted(set(rep))
    log("cells", X.shape[0], "genes", X.shape[1], "samples", {s: int((rep == s).sum()) for s in samples})
    prog_file = f"{a.out}/nmf_programs_all.csv.gz"
    if os.path.exists(prog_file) and not a.smoke:
        df = pd.read_csv(prog_file)
        log("loaded cached programs", len(df))
    else:
        rows = []
        rng = np.random.default_rng(0)
        for s in samples:
            sel = np.where(rep == s)[0]
            if a.max_cells:
                sel = np.sort(rng.choice(sel, min(a.max_cells, len(sel)), replace=False))
            M, gn = sample_matrix(X, genes, sel)
            log(s, "cells", M.shape[0], "genes", M.shape[1])
            run_nmf(M, gn, ks, s, rows)
        df = pd.DataFrame(rows, columns=["sample", "K", "idx", "rank", "gene", "score"])
        df.to_csv(prog_file if not a.smoke else f"{a.out}/smoke_programs.csv.gz", index=False)
    P = to_programs(df)
    R = robust(P, samples)
    pd.DataFrame([dict(sample=p["sample"], K=p["K"], idx=p["idx"], best_overlap_other_sample=p["best_other"], genes=" ".join(p["genes"])) for p in R]).to_csv(
        f"{a.out}/robust_programs{'_smoke' if a.smoke else ''}.csv", index=False)
    mps = make_mps(R)
    pd.DataFrame([dict(MP=i + 1, n_programs=len(m["members"]), samples=",".join(sorted({x[0] for x in m["members"]})),
                       members=";".join(f"{s}_K{k}_{i_}" for s, k, i_ in m["members"]), genes=" ".join(m["genes"])) for i, m in enumerate(mps)]).to_csv(
        f"{a.out}/meta_programs{'_smoke' if a.smoke else ''}.csv", index=False)
    summ = dict(samples=samples, ks=ks, programs_total=len(P), within_ok=sum(p["in_ok"] for p in P), across_ok=sum(p["across_ok"] for p in P),
                both=sum(p["in_ok"] and p["across_ok"] for p in P), robust_nonredundant=len(R),
                robust_per_sample={s: sum(p["sample"] == s for p in R) for s in samples}, n_mps=len(mps),
                mp_sizes=[len(m["members"]) for m in mps])
    json.dump(summ, open(f"{a.out}/summary{'_smoke' if a.smoke else ''}.json", "w"), indent=1)
    log(json.dumps(summ))
