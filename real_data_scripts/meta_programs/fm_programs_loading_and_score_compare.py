#!/usr/bin/env python3
"""Compare the kept (robust, non-redundant) pooled NMF programs of two datasets (default FM01 vs FM06) by
  (1) the FULL loading vectors: the NMF coefficient of every gene shared by the two datasets (Pearson, Spearman, cosine), and
  (2) the CELL SCORES: every program (top-50 genes) is scored in the cells of each dataset (mean z-scored log2(CP10k/10+1) of its genes present there);
      the score of an A-program is correlated with the score of a B-program across the cells of dataset A, and of dataset B.
The programs are refitted with exactly the settings of fm01_meta_programs.py --pool (replicates merged, log2(CP10k/10+1), genes in >= 5 % of cells,
gene-centred, negatives -> 0, NMF nndsvda/cd/max_iter 400/random_state 0, K = 4..9); the refit must reproduce the saved top-50 gene lists of the kept programs (checked).
Output: analysis_data/fm_meta_programs_comparison_pooled/<a>_vs_<b>_program_{loading_correlation,score_correlation_in_<A>,score_correlation_in_<B>}.csv
usage: fm_programs_loading_and_score_compare.py [A B]"""
import sys, time
import numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import spearmanr
from sklearn.decomposition import NMF

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A, B = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("FM01", "FM06")
OUT = f"{ROOT}/analysis_data/fm_meta_programs_comparison_pooled"; KS = [4, 5, 6, 7, 8, 9]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(ds):
    qc = f"{ROOT}/finalized_data/{ds}_data/qc_filtered"
    genes = np.array([l.strip() for l in open(f"{qc}/genes.txt")]); X = sio.mmread(f"{qc}/{ds.lower()}_qc_counts.mtx").tocsr().astype(np.float32)
    return X, genes


def logcp(X, genes, det=0.05):
    cp = X.multiply(1e4 / np.asarray(X.sum(1)).ravel()[:, None]).tocsr(); cp.data = np.log2(cp.data / 10 + 1)
    keep = np.asarray((X > 0).mean(0)).ravel() >= det
    return cp[:, keep].toarray().astype(np.float32), genes[keep]


def fit(ds):
    import os, pickle
    X, genes = load(ds); L, g = logcp(X, genes); del X
    cache = f"{OUT}/.cache_loadings_{ds}.pkl"
    if os.path.exists(cache): H = pickle.load(open(cache, "rb")); log(f"{ds}: loadings from cache")
    else:
        M = L - L.mean(0, keepdims=True); M[M < 0] = 0                                   # exactly as sample_matrix() of fm01_meta_programs.py
        kept = pd.read_csv(f"{ROOT}/analysis_data/{ds.lower()}/data/mp_nmf_pooled/robust_programs.csv"); H = {}
        for k in sorted(set(kept.K)):
            m = NMF(n_components=k, init="nndsvda", solver="cd", max_iter=400, random_state=0).fit(M)
            for r in kept[kept.K == k].itertuples():
                h = m.components_[r.idx]; top = list(g[np.argsort(-h)[:50]]); same = set(top) == set(r.genes.split())
                H[(r.K, r.idx)] = (h, top, same); log(f"{ds} K{r.K}_{r.idx}: refit top-50 identical to the saved kept program: {same}")
        del M; pickle.dump(H, open(cache, "wb"))
    sd = L.std(0); L -= L.mean(0); L /= np.where(sd > 0, sd, 1)                          # z-scored log expression (in place), all genes in >= 5 % of cells
    return dict(g=g, H=H, Z=L.astype(np.float16))


D = {A: fit(A), B: fit(B)}
names = {d: [f"{d}_K{k}_{i}" for (k, i) in D[d]["H"]] for d in D}
assert all(v[2] for d in D for v in D[d]["H"].values()), "refit did not reproduce the saved programs"
shared = sorted(set(D[A]["g"]) & set(D[B]["g"])); log(f"shared genes {len(shared)} ({A}: {len(D[A]['g'])}, {B}: {len(D[B]['g'])})")
ia = pd.Series(np.arange(len(D[A]["g"])), index=D[A]["g"]).loc[shared].to_numpy(); ib = pd.Series(np.arange(len(D[B]["g"])), index=D[B]["g"]).loc[shared].to_numpy()
# (1) full loading vectors over the shared genes
rows = {"pearson": [], "spearman": [], "cosine": []}
for ka, (ha, _, _) in D[A]["H"].items():
    va = ha[ia]; r1, r2, r3 = [], [], []
    for kb, (hb, _, _) in D[B]["H"].items():
        vb = hb[ib]; r1.append(np.corrcoef(va, vb)[0, 1]); r2.append(spearmanr(va, vb)[0]); r3.append(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    rows["pearson"].append(r1); rows["spearman"].append(r2); rows["cosine"].append(r3)
idxA, idxB = [f"{A}_K{k}_{i}" for (k, i) in D[A]["H"]], [f"{B}_K{k}_{i}" for (k, i) in D[B]["H"]]
LD = {m: pd.DataFrame(v, index=idxA, columns=idxB) for m, v in rows.items()}
pd.concat(LD, names=["measure"]).to_csv(f"{OUT}/{A.lower()}_vs_{B.lower()}_program_loading_correlation.csv")
# (2) scores of every program in the cells of each dataset
def score(ds, genes):
    idx = pd.Series(np.arange(len(D[ds]["g"])), index=D[ds]["g"]); j = [idx[g] for g in genes if g in idx.index]; return D[ds]["Z"][:, j].astype(np.float32).mean(1), len(j)
SC = {}
for ds in (A, B):
    S, n_used = {}, {}
    for d2 in (A, B):
        for (k, i), (_, top, _) in D[d2]["H"].items(): S[f"{d2}_K{k}_{i}"], n_used[f"{d2}_K{k}_{i}"] = score(ds, top)
    SC[ds] = (pd.DataFrame(S), n_used)
for ds in (A, B):
    S, n_used = SC[ds]; C = pd.DataFrame(np.corrcoef(S.values.T), index=S.columns, columns=S.columns).loc[idxA, idxB]; C.to_csv(f"{OUT}/{A.lower()}_vs_{B.lower()}_program_score_correlation_in_{ds}.csv")
    SC[ds] = (S, n_used, C)
pd.set_option("display.width", 250)
for m in ("pearson", "spearman"): print(f"\nFULL LOADING VECTORS, {m}, over {len(shared)} shared genes (rows {A}, columns {B})"); print(LD[m].round(2).to_string())
for ds in (A, B): print(f"\nCELL SCORES, Pearson across the cells of {ds} (rows {A} program, columns {B} program; genes of each top-50 present in {ds}: min {min(SC[ds][1].values())})"); print(SC[ds][2].round(2).to_string())
