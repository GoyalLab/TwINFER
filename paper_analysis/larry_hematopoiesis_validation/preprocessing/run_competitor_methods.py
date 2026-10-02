"""Run the competitor GRN-inference methods -- GENIE3, GRNBoost2, PIDC, ppcor, rho, and a random
baseline -- on one gene set's panel, pooled day2+4 cells (the same t1=2/t2=4 window TwINFER is
benchmarked on). Matches yscher's Transcriptomic Distance/helpers/other_grn_methods.py and
helpers/pidc.py methodology exactly:
  - GENIE3 / GRNBoost2: arboreto's own interface, per-target regression on the candidate TFs,
    edge weight = feature importance.
  - ppcor: partial correlation from the precision matrix, p_ij / sqrt(p_ii p_jj), negated.
  - rho: ranked (Spearman) co-expression -- the baseline any network method has to beat.
  - PIDC (Chan, Stumpf & Babtie 2017): specific information / redundancy / unique information /
    proportional-unique-contribution, from the paper's own equations (6-11), Bayesian-blocks
    discretization. Undirected by construction.
  - random: not a network method -- computed at scoring time as the hypergeometric-expected hit
    rate for a random ranking making the same number of calls (see benchmark_competitors.py).

Which gene set: GENE_SET env var, one of the 9 keys in resources/gene_sets.json (default
"correlation_high" for manual/interactive use). TFs = that gene set's own CollecTRI TFs
(resources/gene_sets_detail.json); targets = its full panel (resources/gene_sets.json). Each
method writes resources/networks/{method}_{GENE_SET}.csv (TF, target, importance) -- same schema
as yscher's competitor CSVs, so benchmark_competitors.py reads them all identically.

Run directly: python3 run_competitor_methods.py
Run a different gene set: GENE_SET=variability_low python3 run_competitor_methods.py
Skip the slow ones while testing: METHODS=rho,ppcor python3 run_competitor_methods.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
GENE_SET = os.environ.get("GENE_SET", "correlation_high")
# INPUT_DIR: default twinfer_input (raw counts). INPUT_DIR=twinfer_input_cp10k -> log1p-CP10k
# normalized panels (Spearman / tree GRN methods are not library-normalization invariant).
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] INPUT_CSV = os.path.join(HERE, "resources", os.environ.get("INPUT_DIR", "twinfer_input"), f"{GENE_SET}.csv")
INPUT_CSV = os.path.join(RES_HERE, "resources", os.environ.get("INPUT_DIR", "twinfer_input"), f"{GENE_SET}.csv")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] GENE_SETS_DETAIL = os.path.join(HERE, "resources", "gene_sets_detail.json")
GENE_SETS_DETAIL = os.path.join(RES_HERE, "resources", "gene_sets_detail.json")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] OUT_DIR = os.path.join(HERE, "resources", os.environ.get("NETWORKS_DIR_NAME", "networks"))
OUT_DIR = os.path.join(RES_HERE, "resources", os.environ.get("NETWORKS_DIR_NAME", "networks"))
assert os.path.exists(INPUT_CSV), f"missing input CSV for GENE_SET={GENE_SET!r}: {INPUT_CSV}"
os.makedirs(OUT_DIR, exist_ok=True)

NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("N_CORES", "8")))
PIDC_NSUB = int(os.environ.get("PIDC_CELLS", "8000"))
PIDC_DISC = os.environ.get("PIDC_DISC", "bayesian_blocks")
SEED = 0
METHODS = os.environ.get("METHODS", "rho,ppcor,pidc,genie3,grnboost2").split(",")


def log(m):
    print(m, flush=True)


df = pd.read_csv(INPUT_CSV)
pool = df[df["time_step"].isin([2, 4])].reset_index(drop=True)
gene_cols = [c for c in df.columns if c.endswith("_mRNA")]
keep = [c[:-5] for c in gene_cols]
X_full = pool[gene_cols].to_numpy(dtype=float)
log(f"[{GENE_SET}] pooled day2+4: {X_full.shape[0]:,} cells x {len(keep)} genes")

ALL_GENES = os.environ.get("ALL_GENES", "0") == "1"  # regulators = every panel gene, not just TFs
if ALL_GENES:
    TFS = list(keep)
    log(f"ALL_GENES=1: {len(TFS)} candidate regulators = the full panel (all genes -> all genes)")
else:
    criterion, level = GENE_SET.rsplit("_", 1)  # e.g. "variability_high" -> ("variability", "high")
    detail = json.load(open(GENE_SETS_DETAIL))
    TFS = [tf for tf, _ in detail[criterion][level] if tf in keep]
    log(f"{len(TFS)} candidate regulators (TFs): {TFS}")

GI = {g: i for i, g in enumerate(keep)}
OUT_SUFFIX = "_allgenes" if ALL_GENES else ""


def write(method, rows):
    p = os.path.join(OUT_DIR, f"{method}_{GENE_SET}{OUT_SUFFIX}.csv")
    pd.DataFrame(rows, columns=["TF", "target", "importance"]).sort_values(
        "importance", ascending=False).to_csv(p, index=False)
    log(f"wrote {p} ({len(rows):,} links)")


# ---------------------------------------------------------------- rho: ranked Spearman co-expression
if "rho" in METHODS:
    Rk = np.apply_along_axis(stats.rankdata, 0, X_full)
    Rm = np.corrcoef(Rk, rowvar=False)
    np.fill_diagonal(Rm, 0.0)
    Rm = np.nan_to_num(Rm)
    rows = [(t, g, abs(float(Rm[GI[t], GI[g]]))) for t in TFS for g in keep if g != t]
    write("rho", rows)

# ---------------------------------------------------------------- ppcor: partial correlation
if "ppcor" in METHODS:
    R = np.corrcoef(np.apply_along_axis(stats.rankdata, 0, X_full), rowvar=False)
    R = np.nan_to_num(R)
    np.fill_diagonal(R, 1.0)
    Pm = np.linalg.pinv(R + 1e-8 * np.eye(len(R)))
    d = np.sqrt(np.outer(np.diag(Pm), np.diag(Pm)))
    PC = -Pm / np.where(d > 0, d, 1.0)
    np.fill_diagonal(PC, 0.0)
    n, gsz = X_full.shape
    dof = max(n - gsz - 2, 1)
    rows = [(t, tg, abs(float(PC[GI[t], GI[tg]]))) for t in TFS for tg in keep if tg != t]
    write("ppcor", rows)
    log(f"  ppcor dof={dof}")

# ---------------------------------------------------------------- PIDC (Chan, Stumpf & Babtie 2017)
if "pidc" in METHODS:
    def bayesian_blocks(x, p0=0.05):
        """Scargle et al. 2013 for point measurements; the adaptive binning PIDC recommends."""
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

    def discretise(X, how="bayesian_blocks"):
        out = np.zeros_like(X, dtype=int)
        nb = []
        for j in range(X.shape[1]):
            col = X[:, j]
            e = bayesian_blocks(col) if how == "bayesian_blocks" else np.histogram_bin_edges(col, bins=10)
            e = np.unique(e)
            if len(e) < 2:
                e = np.array([col.min() - .5, col.max() + .5])
            out[:, j] = np.clip(np.digitize(col, e[1:-1]), 0, len(e) - 2)
            nb.append(len(e) - 1)
        return out, np.array(nb)

    def joint(a, b, na, nb):
        t = np.zeros((na, nb))
        np.add.at(t, (a, b), 1.0)
        return t / t.sum()

    def mutual_information(P):
        px = P.sum(1, keepdims=True)
        py = P.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = P * (np.log(P) - np.log(px) - np.log(py))
        return float(np.nansum(np.where(P > 0, v, 0.0)))

    def specific_information(P):
        """I_spec(z;X) for every state z of the SECOND variable (axis 1). Equation 6."""
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

    def pidc_scores(X, genes, how="bayesian_blocks"):
        B, nb = discretise(X, how)
        g = len(genes)
        log(f"  discretisation {how}: bins per gene min {nb.min()} median {int(np.median(nb))} max {nb.max()}")
        MI = np.zeros((g, g))
        JP = {}
        for i in range(g):
            for j in range(i + 1, g):
                P = joint(B[:, i], B[:, j], nb[i], nb[j])
                JP[(i, j)] = P
                MI[i, j] = MI[j, i] = mutual_information(P)
        SI = {}
        for i in range(g):
            for j in range(g):
                if i == j:
                    continue
                P = JP[(i, j)] if i < j else JP[(j, i)].T
                SI[(i, j)] = specific_information(P)
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

    X_pidc = X_full
    if len(X_pidc) > PIDC_NSUB:
        idx = np.random.default_rng(SEED).choice(len(X_pidc), PIDC_NSUB, replace=False)
        X_pidc = X_pidc[idx]
        log(f"  PIDC subsampled to {PIDC_NSUB} cells (Bayesian blocks is O(n^2) per gene)")
    C = pidc_scores(X_pidc, keep, PIDC_DISC)
    rows = [(t, g, float(C[GI[t], GI[g]])) for t in TFS for g in keep if g != t]
    write("pidc", rows)

# ---------------------------------------------------------------- GENIE3 / GRNBoost2
# Reimplemented directly against arboreto.core's exact algorithm (RF_KWARGS/SGBM_KWARGS,
# EarlyStopMonitor, the OOB-heuristic importance de-normalization) rather than calling arboreto
# itself: this installed dask (2025.12.0) has fully removed the legacy dataframe backend
# arboreto's create_graph() requires ("NotImplementedError: The legacy implementation is no
# longer supported"), and even with a from_delayed monkeypatch the distributed compute graph
# hangs indefinitely (verified: plain dask distributed compute works fine on this node, isolating
# the problem to arboreto's own dask-expr-incompatible graph construction, not the cluster).
# At this data scale (44 targets x <=15 TF features) no distributed cluster is needed anyway --
# per-target regression is parallelized directly with joblib.
if "genie3" in METHODS or "grnboost2" in METHODS:
    from joblib import Parallel, delayed
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    RF_KWARGS = {"n_jobs": 1, "n_estimators": 1000, "max_features": "sqrt"}
    SGBM_KWARGS = {"learning_rate": 0.01, "n_estimators": 5000, "max_features": 0.1, "subsample": 0.9}
    EARLY_STOP_WINDOW_LENGTH = 25

    class EarlyStopMonitor:
        """arboreto.core.EarlyStopMonitor, verbatim: stop when the mean OOB improvement over the
        trailing window turns negative."""
        def __init__(self, window_length=EARLY_STOP_WINDOW_LENGTH):
            self.window_length = window_length

        def __call__(self, current_round, regressor, _):
            if current_round >= self.window_length - 1:
                lo = max(0, current_round - self.window_length + 1)
                hi = current_round + 1
                return np.mean(regressor.oob_improvement_[lo:hi]) < 0
            return False

    def _fit_one_target(target, regressor_cls, kwargs, with_early_stop):
        tf_names = [t for t in TFS if t != target]  # arboreto.core.clean(): drop self if target is also a TF
        Xtf = X_full[:, [GI[t] for t in tf_names]]
        y = X_full[:, GI[target]]
        reg = regressor_cls(random_state=SEED, **kwargs)
        if with_early_stop:
            reg.fit(Xtf, y, monitor=EarlyStopMonitor())
            importances = reg.feature_importances_ * len(reg.estimators_)  # de-normalize OOB heuristic
        else:
            reg.fit(Xtf, y)
            importances = reg.feature_importances_
        return [(tf, target, float(imp)) for tf, imp in zip(tf_names, importances) if imp > 0]

    if "genie3" in METHODS:
        t0 = time.time()
        out = Parallel(n_jobs=NCPU)(
            delayed(_fit_one_target)(g, RandomForestRegressor, RF_KWARGS, False) for g in keep)
        rows = [r for sub in out for r in sub]
        write("genie3", rows)
        log(f"  genie3: {__import__('time').time() - t0:.1f}s for {len(keep)} targets")

    if "grnboost2" in METHODS:
        t0 = time.time()
        out = Parallel(n_jobs=NCPU)(
            delayed(_fit_one_target)(g, GradientBoostingRegressor, SGBM_KWARGS, True) for g in keep)
        rows = [r for sub in out for r in sub]
        write("grnboost2", rows)
        log(f"  grnboost2: {__import__('time').time() - t0:.1f}s for {len(keep)} targets")

log("\ndone")
