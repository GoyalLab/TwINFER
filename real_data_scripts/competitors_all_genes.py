#!/usr/bin/env python3
"""
competitors_all_genes.py

Competitor GRN methods on the PDF's universe: every gene detected in >=5% of cells is a candidate target, regulators are the
expressed TFs (the all_tfs.txt of gene_modules/), every TF->gene pair is scored. Input is the log-normalised matrix already
built for the module runs (genes x cells CSV, all cells).

Methods (implementations follow paper_analysis/fatemap_pipeline/run_competitors_fatemap.py):
  rho        |Spearman| of every TF with every gene
  ppcor      |partial Spearman| from the pseudo-inverse of the rank-correlation matrix of ALL genes (ridge 1e-8, as the pipeline).
             NOTE: with ~10k genes and ~10k cells the correlation matrix is close to singular, so this is ill-conditioned.
  genie3     RandomForest (1000 trees, max_features sqrt), one model per target, regulators = TFs
  grnboost2  GradientBoosting (lr 0.01, <=5000 trees, subsample 0.9, max_features 0.1, early stop), one model per target
PIDC is not run (cost ~ genes^3 in pure Python; infeasible at this size).

Modes:  run   --method M [--chunk i --n-chunks n] [--max-targets k]   writes <out>/<M>_chunk<i>.npz (importance matrix TF x targets)
        eval  merges chunks and reports AUPRC x base rate (1.0 = chance) vs CollecTRI TF->target edges.
"""
import argparse, os, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import auc, precision_recall_curve

SEED = 0
RF_KWARGS = {"n_jobs": 1, "n_estimators": 1000, "max_features": "sqrt"}
SGBM_KWARGS = {"learning_rate": 0.01, "n_estimators": 5000, "max_features": 0.1, "subsample": 0.9}
ES_WINDOW = 25
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(expr_csv, tfs_file):
    genes, blocks = [], []
    for ch in pd.read_csv(expr_csv, index_col=0, chunksize=1000):       # chunked, float32: stays small on the login node
        genes += list(ch.index)
        blocks.append(ch.to_numpy(np.float32))
    X = np.vstack(blocks).T.copy()                                       # cells x genes
    tfs = [t for t in (l.strip() for l in open(tfs_file)) if t]
    gi = {g: i for i, g in enumerate(genes)}
    tfs = [t for t in dict.fromkeys(tfs) if t in gi]
    return X, genes, tfs, np.array([gi[t] for t in tfs])


def ranks(X):
    return np.apply_along_axis(rankdata, 0, X).astype(np.float32)


def corr_matrix(Ra, Rb):
    A = (Ra - Ra.mean(0)) / (Ra.std(0) + 1e-12)
    B = (Rb - Rb.mean(0)) / (Rb.std(0) + 1e-12)
    return (A.T @ B) / Ra.shape[0]


class EarlyStop:
    def __call__(self, i, reg, _):
        if i >= ES_WINDOW - 1:
            return np.mean(reg.oob_improvement_[max(0, i - ES_WINDOW + 1):i + 1]) < 0
        return False


def fit_target(Xtf, tf_gene, ti, y, method):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    keep = np.flatnonzero(tf_gene != ti)                                 # a gene never regulates itself
    out = np.zeros(Xtf.shape[1], dtype=np.float32)
    if method == "genie3":
        r = RandomForestRegressor(random_state=SEED, **RF_KWARGS).fit(Xtf[:, keep], y)
        out[keep] = r.feature_importances_
    else:
        r = GradientBoostingRegressor(random_state=SEED, **SGBM_KWARGS).fit(Xtf[:, keep], y, monitor=EarlyStop())
        out[keep] = r.feature_importances_ * len(r.estimators_)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "eval"])
    ap.add_argument("--expr", required=True)
    ap.add_argument("--tfs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["rho", "ppcor", "genie3", "grnboost2"])
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--max-targets", type=int, default=None, help="timing/testing: only the first k targets of the chunk")
    ap.add_argument("--collectri", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, genes, tfs, tf_gene = load(a.expr, a.tfs)
    G, T = len(genes), len(tfs)
    log(f"{X.shape[0]:,} cells x {G:,} genes, {T} TF regulators")

    if a.mode == "run":
        if a.method in ("rho", "ppcor"):
            R = ranks(X)
            if a.method == "rho":
                M = np.abs(corr_matrix(R[:, tf_gene], R)).astype(np.float32)          # TF x gene
            else:
                C = np.nan_to_num(np.corrcoef(R, rowvar=False)); np.fill_diagonal(C, 1.0)
                P = np.linalg.pinv(C + 1e-8 * np.eye(G)); d = np.sqrt(np.outer(np.diag(P), np.diag(P)))
                PC = -P / np.where(d > 0, d, 1.0); np.fill_diagonal(PC, 0.0)
                M = np.abs(PC[tf_gene]).astype(np.float32)
            M[np.arange(T), tf_gene] = 0.0
            np.savez_compressed(f"{a.out}/{a.method}.npz", M=M, tfs=tfs, genes=genes)
            log(f"{a.method} done")
        else:
            from joblib import Parallel, delayed
            Xtf = np.ascontiguousarray(X[:, tf_gene])
            tgt = np.array_split(np.arange(G), a.n_chunks)[a.chunk]
            if a.max_targets:
                tgt = np.random.default_rng(SEED).choice(tgt, min(a.max_targets, len(tgt)), replace=False)
            t0 = time.time()
            res = Parallel(n_jobs=a.n_jobs)(delayed(fit_target)(Xtf, tf_gene, int(t), X[:, t], a.method) for t in tgt)
            dt = time.time() - t0
            log(f"{a.method}: {len(tgt)} targets in {dt:.0f}s = {dt / len(tgt):.1f}s/target wall with {a.n_jobs} workers "
                f"-> all {G:,} targets ~ {dt / len(tgt) * G / 3600:.1f} h on this node")
            tag = "" if not a.max_targets else "_timing"
            np.savez_compressed(f"{a.out}/{a.method}{tag}_chunk{a.chunk}.npz", M=np.stack(res, 1), targets=tgt, tfs=tfs, genes=genes)
    else:
        ct = pd.read_csv(a.collectri, sep="\t")
        edges = set(zip(ct.source_genesymbol, ct.target_genesymbol))
        Y = np.zeros((T, G), dtype=np.int8)
        gi = {g: i for i, g in enumerate(genes)}
        for ti, t in enumerate(tfs):
            for tg in (g for (s, g) in edges if s == t and g in gi and g != t):
                Y[ti, gi[tg]] = 1
        valid = np.ones((T, G), bool); valid[np.arange(T), tf_gene] = False
        rows = []
        for m in ("rho", "ppcor", "genie3", "grnboost2"):
            if m in ("rho", "ppcor"):
                p = f"{a.out}/{m}.npz"
                if not os.path.exists(p): continue
                M = np.load(p)["M"]
            else:
                fs = sorted(f for f in os.listdir(a.out) if f.startswith(f"{m}_chunk") and "timing" not in f)
                if not fs: continue
                M = np.zeros((T, G), np.float32); cov = np.zeros(G, bool)
                for f in fs:
                    z = np.load(f"{a.out}/{f}", allow_pickle=True); M[:, z["targets"]] = z["M"]; cov[z["targets"]] = True
                if not cov.all():
                    log(f"{m}: only {cov.sum()}/{G} targets finished; evaluating those"); valid &= cov[None, :]
            sc, y = M[valid], Y[valid]
            p_, r_, _ = precision_recall_curve(y, sc)
            rows.append(dict(method=m, auprc_x_base=auc(r_, p_) / y.mean(), n_edges=int(y.sum()), n_pairs=len(y), base_rate=float(y.mean())))
        R = pd.DataFrame(rows); R.to_csv(f"{a.out}/competitors_auprc.csv", index=False); print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
