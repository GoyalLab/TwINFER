#!/usr/bin/env python3
"""Are the NMF programs inherited? Clone-mate correlation (TwinScore sister correlation h) of each program's score.

For each robust program of a dataset (analysis_data/<fm>/data/mp_nmf/robust_programs.csv, top-50 genes):
  score(cell) = mean over the program's genes of the gene's z-score (z over all cells of the dataset, clipped to +-5)
                of log2(CP10k/10 + 1)
  h = TwinScore sister correlation C(x,x) exactly as TwinScore_procedure_and_top100.pdf sec. 2 and 5:
      clone-weighted mid-ranks of the score over ALL cells (cell weight 1/n_c); within-clone cell pairs with weight 1/P_c;
      ranks centred over twin cells, h = sum_p w_p r_a r_b / M, M = sum_p w_p (r_a^2 + r_b^2)/2
      analytic z = h / sigma_xx, sigma_xx^2 = 1/m_t, m_t = N_t^2 / sum_c P_c^{-1}  (sec. 6)
      permutation p: the scores are permuted over cells (clone structure and weights fixed), --nperm draws
References computed the same way:
  technical cell covariates: log total counts, log n genes, % mitochondrial reads (obs_metadata.csv)
  random gene sets: --nrand sets of 50 genes, each gene drawn from the same mean-expression decile as a program gene
Clones = fatemap_clone_singletcode of obs_metadata.csv; cells without a clone are singletons (weight 1, in no pair).
usage: FM_DATASET=FM01 program_heritability.py [--nperm 200] [--nrand 100]
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
DS = os.environ.get("FM_DATASET", "FM01")
QC = f"{ROOT}/finalized_data/{DS}_data/qc_filtered"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def wranks(v, w):
    """clone-weighted mid-ranks: sum of weights below + half the weight of ties"""
    u, inv = np.unique(v, return_inverse=True)
    ws = np.bincount(inv, weights=w, minlength=len(u))
    below = np.cumsum(ws) - ws
    return (below + 0.5 * ws)[inv]


class Twin:
    def __init__(self, clone):
        clone = pd.Series(clone).astype("object").to_numpy()
        n = len(clone); self.n = n
        cs = pd.Series(clone); cnt = cs.map(cs.value_counts())
        nc = np.where(cs.isna(), 1, cnt.fillna(1)).astype(float)
        self.w = 1.0 / nc
        a, b, wp = [], [], []; invP = 0.0; self.Nt = 0
        for c, idx in pd.Series(np.arange(n)).groupby(cs.to_numpy()).groups.items():
            idx = np.asarray(idx)
            if len(idx) < 2: continue
            ia, ib = np.triu_indices(len(idx), 1); P = len(ia)
            a.append(idx[ia]); b.append(idx[ib]); wp.append(np.full(P, 1.0 / P)); invP += 1.0 / P; self.Nt += 1
        self.a = np.concatenate(a); self.b = np.concatenate(b); self.wp = np.concatenate(wp)
        self.mt = self.Nt ** 2 / invP; self.sig = 1.0 / np.sqrt(self.mt)

    def h(self, v):
        R = wranks(v, self.w)
        mu = (self.wp * (R[self.a] + R[self.b]) / 2).sum() / self.Nt
        ra, rb = R[self.a] - mu, R[self.b] - mu
        M = (self.wp * (ra ** 2 + rb ** 2) / 2).sum()
        return (self.wp * ra * rb).sum() / M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nperm", type=int, default=200); ap.add_argument("--nrand", type=int, default=100)
    ap.add_argument("--by-replicate", action="store_true", help="also run each replicate on its own (diagnostic only; drops clone mates split across A and B)")
    a = ap.parse_args(); rng = np.random.default_rng(0)
    out =f"{ROOT}/analysis_data/{DS.lower()}/data/mp_program_heritability"; os.makedirs(out, exist_ok=True)
    genes = np.array([l.strip() for l in open(f"{QC}/genes.txt")]); gi = {g: i for i, g in enumerate(genes)}
    cells = np.array([l.strip() for l in open(f"{QC}/cells.txt")])
    obs = pd.read_csv(f"{QC}/obs_metadata.csv", index_col=0).loc[cells]
    X = sio.mmread(f"{QC}/{DS.lower()}_qc_counts.mtx").tocsc().astype(np.float32)
    tot = np.asarray(X.sum(1)).ravel(); X = X.tocsr(); X = X.multiply(1e4 / tot[:, None]).tocsc(); X.data = np.log2(X.data / 10 + 1)
    det = np.asarray((X > 0).mean(0)).ravel(); meanx = np.asarray(X.mean(0)).ravel(); ok = np.where(det >= 0.05)[0]
    R = pd.read_csv(f"{ROOT}/analysis_data/{DS.lower()}/data/mp_nmf/robust_programs.csv")
    q = pd.qcut(meanx[ok], 10, labels=False, duplicates="drop"); dec = dict(zip(ok, q)); bins = {d: ok[q == d] for d in np.unique(q)}
    need = set(); progs = []
    for r in R.itertuples():
        g = [gi[x] for x in r.genes.split() if x in gi and det[gi[x]] >= 0.05]
        sets = [[int(rng.choice(bins[dec[i]])) for i in g] for _ in range(a.nrand)]
        progs.append((r, g, sets)); need.update(g); [need.update(s_) for s_ in sets]
    need = np.array(sorted(need)); pos = {g: k for k, g in enumerate(need)}
    D0 = X[:, need].toarray(); log("dense matrix", D0.shape)
    # strata: all cells of the dataset ("ALL", as the TwinScore runs) and each replicate on its own (ranks, z-scores and clone pairs inside the replicate)
    # Default = all cells together: A and B hold sister cells of the same clones (clones span replicates), so splitting by replicate would drop real clone mates.
    # --by-replicate adds the per-replicate strata only as a batch diagnostic.
    strata = [("ALL", np.arange(len(cells)))]
    if a.by_replicate:
        strata += [(r_, np.where(obs["replicate"].to_numpy() == r_)[0]) for r_ in sorted(obs["replicate"].unique())]
    rows = []; des = []
    for st, ix in strata:
        sub = D0[ix]; sd = sub.std(0); D = np.clip((sub - sub.mean(0)) / np.where(sd > 0, sd, 1), -5, 5).astype(np.float32)
        T = Twin(obs["fatemap_clone_singletcode"].iloc[ix].to_numpy())
        log(DS, st, "cells", len(ix), "clones >=2 cells N_t", T.Nt, "twin pairs", len(T.a), "sigma_xx", round(T.sig, 4))
        def score(idx): return D[:, [pos[i] for i in idx]].mean(1)
        def perm_p(v, h0):
            c = 0
            for _ in range(a.nperm): c += T.h(rng.permutation(v)) >= h0
            return (1 + c) / (a.nperm + 1)
        o = obs.iloc[ix]; ref = {}
        for nm, v in (("log_total_counts", np.log(o["total_counts"].to_numpy())), ("log_n_genes", np.log(o["n_genes"].to_numpy())), ("pct_mito", o["pct_counts_mt"].to_numpy())):
            ref[nm] = T.h(v)
        log(st, "technical references h:", {k: round(v, 3) for k, v in ref.items()})
        des.append(dict(dataset=DS, stratum=st, n_cells=len(ix), N_t=T.Nt, n_twin_pairs=len(T.a), m_t=T.mt, sigma_xx=T.sig, nperm=a.nperm, nrand=a.nrand, **{f"h_{k}": v for k, v in ref.items()}))
        for r, g, sets in progs:
            v = score(g); h0 = T.h(v); hr = np.array([T.h(score(s_)) for s_ in sets])
            rows.append(dict(dataset=DS, stratum=st, replicate=r.sample, K=r.K, idx=r.idx, n_genes_used=len(g), top6=" ".join(r.genes.split()[:6]), h=h0, z_analytic=h0 / T.sig,
                             p_perm=perm_p(v, h0), random_h_median=np.median(hr), random_h_p95=np.percentile(hr, 95), frac_random_ge_h=float((hr >= h0).mean()),
                             h_log_total_counts=ref["log_total_counts"], h_log_n_genes=ref["log_n_genes"], h_pct_mito=ref["pct_mito"]))
        log(DS, st, "done", len(progs), "programs")
    pd.DataFrame(rows).to_csv(f"{out}/program_heritability.csv", index=False); pd.DataFrame(des).to_csv(f"{out}/design_and_technical_references.csv", index=False)


if __name__ == "__main__":
    main()
