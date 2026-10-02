#!/usr/bin/env python3
"""Ports preprocessing/run_competitor_methods.py (rho, ppcor, PIDC, GENIE3, GRNBoost2) to
FM06/FM08, run in "ALL_GENES" mode (every panel gene is a candidate regulator, not just
CollecTRI TFs) so the output covers the SAME ordered-pair universe as
run_infer_fatemap.py's ranked_edges_*.csv / apply_todo4v2_vs_collectri.py's evaluation --
letting the two be compared on identical pairs, same as the LARRY reference's *_allgenes.csv.

Reuses the already-built twinfer_input_{label}_{geneset}.csv (log1p(CP10k) expression,
run_infer_fatemap.py's output) as input -- same cells/genes TwINFER itself was run on.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

SEED = 0
PIDC_NSUB = 8000
PIDC_DISC = "bayesian_blocks"


def log(m):
    print(m, flush=True)


def load_panel(dataset, gene_set):
    label = dataset.lower()
    path = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/twinfer_input_{label}_{gene_set}.csv"
    return load_panel_from_csv(path, dataset, gene_set)


def load_panel_from_csv(path, dataset, gene_set):
    df = pd.read_csv(path)
    gene_cols = [c for c in df.columns if c.endswith("_mRNA")]
    keep = [c[:-5] for c in gene_cols]
    X = df[gene_cols].to_numpy(dtype=float)
    log(f"[{dataset}/{gene_set}] {X.shape[0]:,} cells x {len(keep)} genes (ALL_GENES mode, {path})")
    return X, keep


OUT_TAG = ""  # "_absplit" when the panel is the A/B-split input (see --absplit)


def write(out_dir, method, gs, rows):
    p = os.path.join(out_dir, f"{method}_{gs}_allgenes{OUT_TAG}.csv")
    pd.DataFrame(rows, columns=["TF", "target", "importance"]).sort_values(
        "importance", ascending=False).to_csv(p, index=False)
    log(f"wrote {p} ({len(rows):,} links)")


def run_rho(X, keep, GI, TFS, out_dir, gs):
    Rk = np.apply_along_axis(stats.rankdata, 0, X)
    Rm = np.corrcoef(Rk, rowvar=False)
    np.fill_diagonal(Rm, 0.0)
    Rm = np.nan_to_num(Rm)
    rows = [(t, g, abs(float(Rm[GI[t], GI[g]]))) for t in TFS for g in keep if g != t]
    write(out_dir, "rho", gs, rows)


def run_ppcor(X, keep, GI, TFS, out_dir, gs):
    R = np.corrcoef(np.apply_along_axis(stats.rankdata, 0, X), rowvar=False)
    R = np.nan_to_num(R)
    np.fill_diagonal(R, 1.0)
    Pm = np.linalg.pinv(R + 1e-8 * np.eye(len(R)))
    d = np.sqrt(np.outer(np.diag(Pm), np.diag(Pm)))
    PC = -Pm / np.where(d > 0, d, 1.0)
    np.fill_diagonal(PC, 0.0)
    rows = [(t, tg, abs(float(PC[GI[t], GI[tg]]))) for t in TFS for tg in keep if tg != t]
    write(out_dir, "ppcor", gs, rows)


def _bayesian_blocks(x, p0=0.05):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 3 or np.all(x == x[0]):
        return np.array([x[0] - 0.5, x[0] + 0.5])
    edges = np.concatenate([x[:1], 0.5 * (x[1:] + x[:-1]), x[-1:]])
    block_length = x[-1] - edges
    best = np.zeros(n, dtype=float)
    last = np.zeros(n, dtype=int)
    ncp_prior = 4 - np.log(73.53 * p0 * n ** -0.478)
    for k in range(n):
        width = block_length[:k + 1] - block_length[k + 1]
        count = np.arange(k + 1, 0, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fit = count * (np.log(count) - np.log(np.where(width > 0, width, np.nan)))
        fit = np.nan_to_num(fit, nan=-np.inf, neginf=-np.inf)
        A = fit - ncp_prior
        A[1:] += best[:k]
        i = int(np.argmax(A))
        last[k] = i
        best[k] = A[i]
    ch = []
    i = n
    while i > 0:
        ch.append(i)
        i = last[i - 1]
    return edges[np.sort(np.array(ch + [0]))]


def _discretise(X, how="bayesian_blocks"):
    out = np.zeros_like(X, dtype=int)
    nb = []
    for j in range(X.shape[1]):
        col = X[:, j]
        e = _bayesian_blocks(col) if how == "bayesian_blocks" else np.histogram_bin_edges(col, bins=10)
        e = np.unique(e)
        if len(e) < 2:
            e = np.array([col.min() - .5, col.max() + .5])
        out[:, j] = np.clip(np.digitize(col, e[1:-1]), 0, len(e) - 2)
        nb.append(len(e) - 1)
    return out, np.array(nb)


def _joint(a, b, na, nb):
    t = np.zeros((na, nb))
    np.add.at(t, (a, b), 1.0)
    return t / t.sum()


def _mutual_information(P):
    px = P.sum(1, keepdims=True)
    py = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = P * (np.log(P) - np.log(px) - np.log(py))
    return float(np.nansum(np.where(P > 0, v, 0.0)))


def _specific_information(P):
    pz = P.sum(0)
    px = P.sum(1)
    out = np.zeros(P.shape[1])
    for zi in range(P.shape[1]):
        if pz[zi] <= 0:
            continue
        pxz = P[:, zi] / pz[zi]
        with np.errstate(divide="ignore", invalid="ignore"):
            pzx = np.where(px > 0, P[:, zi] / px, 0.0)
            term = np.where(pzx > 0, np.log(1.0 / pz[zi]) - np.log(1.0 / pzx), 0.0)
        out[zi] = float(np.nansum(pxz * term))
    return out, pz


def _pidc_scores(X, genes, how="bayesian_blocks"):
    B, nb = _discretise(X, how)
    g = len(genes)
    log(f"  discretisation {how}: bins per gene min {nb.min()} median {int(np.median(nb))} max {nb.max()}")
    MI = np.zeros((g, g))
    JP = {}
    for i in range(g):
        for j in range(i + 1, g):
            P = _joint(B[:, i], B[:, j], nb[i], nb[j])
            JP[(i, j)] = P
            MI[i, j] = MI[j, i] = _mutual_information(P)
    SI = {}
    for i in range(g):
        for j in range(g):
            if i == j:
                continue
            P = JP[(i, j)] if i < j else JP[(j, i)].T
            SI[(i, j)] = _specific_information(P)
    PUC = np.zeros((g, g))
    for i in range(g):
        for j in range(g):
            if i == j:
                continue
            mij = MI[i, j]
            if mij <= 0:
                continue
            tot = 0.0
            for k in range(g):
                if k == i or k == j:
                    continue
                si_i, pz = SI[(i, j)]
                si_k, _ = SI[(k, j)]
                red = float(np.sum(pz * np.minimum(si_i, si_k)))
                tot += (mij - red) / mij
            PUC[i, j] = tot
    C = np.zeros((g, g))
    for i in range(g):
        v = PUC[i, np.arange(g) != i]
        v = v[np.isfinite(v)]
        try:
            a, loc, sc = stats.gamma.fit(v[v > 0], floc=0)
            F = lambda u, a=a, sc=sc: stats.gamma.cdf(u, a, loc=0, scale=sc)
        except Exception:
            mu, sd = float(np.mean(v)), float(np.std(v) or 1.0)
            F = lambda u, mu=mu, sd=sd: stats.norm.cdf(u, mu, sd)
        for j in range(g):
            if i != j:
                C[i, j] += F(PUC[i, j])
    return C


def run_pidc(X, keep, GI, TFS, out_dir, gs):
    X_pidc = X
    if len(X_pidc) > PIDC_NSUB:
        idx = np.random.default_rng(SEED).choice(len(X_pidc), PIDC_NSUB, replace=False)
        X_pidc = X_pidc[idx]
        log(f"  PIDC subsampled to {PIDC_NSUB} cells")
    C = _pidc_scores(X_pidc, keep, PIDC_DISC)
    rows = [(t, g, float(C[GI[t], GI[g]])) for t in TFS for g in keep if g != t]
    write(out_dir, "pidc", gs, rows)


RF_KWARGS = {"n_jobs": 1, "n_estimators": 1000, "max_features": "sqrt"}
SGBM_KWARGS = {"learning_rate": 0.01, "n_estimators": 5000, "max_features": 0.1, "subsample": 0.9}
EARLY_STOP_WINDOW_LENGTH = 25


class EarlyStopMonitor:
    def __init__(self, window_length=EARLY_STOP_WINDOW_LENGTH):
        self.window_length = window_length

    def __call__(self, current_round, regressor, _):
        if current_round >= self.window_length - 1:
            lo = max(0, current_round - self.window_length + 1)
            hi = current_round + 1
            return np.mean(regressor.oob_improvement_[lo:hi]) < 0
        return False


def _fit_one_target(X, GI, TFS, target, regressor_cls, kwargs, with_early_stop):
    tf_names = [t for t in TFS if t != target]
    Xtf = X[:, [GI[t] for t in tf_names]]
    y = X[:, GI[target]]
    reg = regressor_cls(random_state=SEED, **kwargs)
    if with_early_stop:
        reg.fit(Xtf, y, monitor=EarlyStopMonitor())
        importances = reg.feature_importances_ * len(reg.estimators_)
    else:
        reg.fit(Xtf, y)
        importances = reg.feature_importances_
    return [(tf, target, float(imp)) for tf, imp in zip(tf_names, importances) if imp > 0]


def run_tree_method(X, keep, GI, TFS, out_dir, gs, method, n_cores):
    from joblib import Parallel, delayed
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    t0 = time.time()
    if method == "genie3":
        regressor_cls, kwargs, early = RandomForestRegressor, RF_KWARGS, False
    else:
        regressor_cls, kwargs, early = GradientBoostingRegressor, SGBM_KWARGS, True
    out = Parallel(n_jobs=n_cores)(
        delayed(_fit_one_target)(X, GI, TFS, g, regressor_cls, kwargs, early) for g in keep)
    rows = [r for sub in out for r in sub]
    write(out_dir, method, gs, rows)
    log(f"  {method}: {time.time()-t0:.1f}s for {len(keep)} targets")


def main():
    ap = argparse.ArgumentParser()
    # 2026-09-22: added SPACEBAR -- load_panel/write are dataset-agnostic (just need
    # twinfer_input_{label}_{gene_set}.csv to exist, which run_infer_spacebar.py already
    # produces), only this choices list was FM06/FM08-specific.
    # 2026-09-23: added FM01 and the per-stage Watermelon datasets.
    # 2026-09-29: added HPSC_20260927 (endo_T0/endo_T1 two-timepoint hESC->endoderm data).
    ap.add_argument("dataset", choices=["FM01", "FM06", "FM08", "SPACEBAR", "HS054",
                                        "Watermelon_naive", "Watermelon_lag", "Watermelon_late",
                                        "HPSC_20260927"])
    ap.add_argument("--gene-set", required=True)
    ap.add_argument("--methods", default="rho,ppcor,pidc,genie3,grnboost2")
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--absplit", action="store_true",
                    help="use the EXACT cells/genes of the A/B-split TwINFER run (build_twinfer_input_ab, incl. any per-side "
                         "subsampling) instead of the pooled twinfer_input csv; outputs tagged _absplit")
    ap.add_argument("--paper", action="store_true",
                    help="use the EXACT cells/genes of twinscore_paper.py's single-replicate, clone-filtered panel "
                         "(twinfer_input_{label}_{gs}_paper.csv, written by that script); outputs tagged _paper")
    args = ap.parse_args()
    global OUT_TAG

    label = args.dataset.lower()
    out_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/networks"
    os.makedirs(out_dir, exist_ok=True)

    if args.paper:
        OUT_TAG = "_paper"
        path = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/twinfer_input_{label}_{args.gene_set}_paper.csv"
        X, keep = load_panel_from_csv(path, args.dataset, args.gene_set)
    elif args.absplit:
        OUT_TAG = "_absplit"
        # [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
        # sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from paper_analysis.fatemap_pipeline.run_infer_fatemap_ab_split import build_twinfer_input_ab
        gene_set = json.load(open(f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/gene_sets_{label}.json"))[args.gene_set]
        df = build_twinfer_input_ab(args.dataset, gene_set)
        keep = list(gene_set)
        X = df[[f"{g}_mRNA" for g in keep]].to_numpy(dtype=float)
        log(f"[{args.dataset}/{args.gene_set}] A/B-split input: {X.shape[0]:,} cells x {len(keep)} genes (ALL_GENES mode)")
    elif not args.paper:
        X, keep = load_panel(args.dataset, args.gene_set)
    TFS = list(keep)  # ALL_GENES mode
    GI = {g: i for i, g in enumerate(keep)}
    methods = args.methods.split(",")

    if "rho" in methods:
        run_rho(X, keep, GI, TFS, out_dir, args.gene_set)
    if "ppcor" in methods:
        run_ppcor(X, keep, GI, TFS, out_dir, args.gene_set)
    if "pidc" in methods:
        run_pidc(X, keep, GI, TFS, out_dir, args.gene_set)
    if "genie3" in methods:
        run_tree_method(X, keep, GI, TFS, out_dir, args.gene_set, "genie3", args.n_cores)
    if "grnboost2" in methods:
        run_tree_method(X, keep, GI, TFS, out_dir, args.gene_set, "grnboost2", args.n_cores)

    log("done")


if __name__ == "__main__":
    main()
