# [copied 2026-09-30 from /gpfs/projects/b1255/yscher/Transcriptomic Distance/helpers/pidc.py (author/owner: yscher/laj2116), md5 b823274e8bd0427ce355cb42d517f6db; unmodified below this header]
"""PIDC (Chan, Stumpf & Babtie 2017, Cell Systems), implemented from the paper's own equations.

  specific information   I_spec(z;X) = sum_x p(x|z) [ log(1/p(z)) - log(1/p(z|x)) ]        (eq 6)
  redundancy             Red(Z;X,Y)  = sum_z p(z) min_{S in {X,Y}} I_spec(z;S)              (eq 7)
  unique information     Unique_Y(Z;X) = I(X;Z) - Red(Z;X,Y)                                (eq 8)
  proportional unique
  contribution           u_XY = sum_{Z != X,Y} [ Unique_Z(X;Y) + Unique_Z(Y;X) ] / I(X;Y)   (eq 10)
  edge confidence        c = F_X(u_XY) + F_Y(u_XY), F the fitted CDF of that gene's PUC     (eq 11)

DISCRETISATION: Bayesian blocks, the adaptive variable-width algorithm the authors recommend, with
the maximum-likelihood estimator for the probability tables. A uniform-width fallback is available
via PIDC_DISC=uniform for speed, and which was used is printed, because the discretisation changes
the mutual information estimates and therefore the result.

UNDIRECTED. PIDC scores a gene PAIR; it does not orient. Both orientations present in the universe
receive the same score, which shows up as called_both in the orientation scoring. That is a property
of the method.
"""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from dataset_paths import dpath
import provenance as _prov
import day_column as _dc  # one timepoint reading for every generator; see helpers/day_column.py

def log(*a): print(*a, flush=True)

def bayesian_blocks(x, p0=0.05):
    """Scargle et al. 2013 for point measurements; the adaptive binning PIDC recommends."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 3 or np.all(x == x[0]): return np.array([x[0] - 0.5, x[0] + 0.5])
    edges = np.concatenate([x[:1], 0.5 * (x[1:] + x[:-1]), x[-1:]])
    block_length = x[-1] - edges
    best = np.zeros(n, dtype=float); last = np.zeros(n, dtype=int)
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
        last[k] = i; best[k] = A[i]
    ch = []; i = n
    while i > 0:
        ch.append(i); i = last[i - 1]
    return edges[np.sort(np.array(ch + [0]))]

def discretise(X, how="bayesian_blocks"):
    out = np.zeros_like(X, dtype=int); nb = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if how == "bayesian_blocks":
            e = bayesian_blocks(col)
        else:
            e = np.histogram_bin_edges(col, bins=10)
        e = np.unique(e)
        if len(e) < 2: e = np.array([col.min() - .5, col.max() + .5])
        out[:, j] = np.clip(np.digitize(col, e[1:-1]), 0, len(e) - 2)
        nb.append(len(e) - 1)
    return out, np.array(nb)

def joint(a, b, na, nb):
    t = np.zeros((na, nb)); np.add.at(t, (a, b), 1.0); return t / t.sum()

def mutual_information(P):
    px = P.sum(1, keepdims=True); py = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = P * (np.log(P) - np.log(px) - np.log(py))
    return float(np.nansum(np.where(P > 0, v, 0.0)))

def specific_information(P):
    """I_spec(z;X) for every state z of the SECOND variable (axis 1). Equation 6."""
    pz = P.sum(0)                      # p(z)
    px = P.sum(1)                      # p(x)
    out = np.zeros(P.shape[1])
    for zi in range(P.shape[1]):
        if pz[zi] <= 0: continue
        pxz = P[:, zi] / pz[zi]        # p(x|z)
        with np.errstate(divide="ignore", invalid="ignore"):
            pzx = np.where(px > 0, P[:, zi] / px, 0.0)     # p(z|x)
            term = np.where(pzx > 0, np.log(1.0 / pz[zi]) - np.log(1.0 / pzx), 0.0)
        out[zi] = float(np.nansum(pxz * term))
    return out, pz

def pidc_scores(X, genes, how="bayesian_blocks"):
    B, nb = discretise(X, how)
    g = len(genes)
    log(f"  discretisation {how}: bins per gene min {nb.min()} median {int(np.median(nb))} max {nb.max()}")
    MI = np.zeros((g, g)); JP = {}
    for i in range(g):
        for j in range(i + 1, g):
            P = joint(B[:, i], B[:, j], nb[i], nb[j])
            JP[(i, j)] = P
            MI[i, j] = MI[j, i] = mutual_information(P)
    # specific information of X about each state of Z, for every ordered pair
    SI = {}
    for i in range(g):
        for j in range(g):
            if i == j: continue
            P = JP[(i, j)] if i < j else JP[(j, i)].T
            SI[(i, j)] = specific_information(P)     # I_spec(z_j ; X_i) over states of j
    PUC = np.zeros((g, g))
    for i in range(g):
        for j in range(g):
            if i == j: continue
            mij = MI[i, j]
            if mij <= 0: continue
            tot = 0.0
            for k in range(g):
                if k == i or k == j: continue
                # Unique_k(i;j): I(i;j) - Red(j; i,k) with the target being j
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
            if i != j: C[i, j] += F(PUC[i, j])
    return C + C.T if False else C            # c = F_X(u) + F_Y(u): both halves already added

if __name__ == "__main__":
    PANEL = open(os.path.join(ROOT, os.environ["PANEL_FILE"])).read().strip().split(",")
    TFS = [t for t in open(os.path.join(ROOT, os.environ["TF_LIST_FILE"])).read().strip().split(",") if t in PANEL]
    MATRIX = os.environ["MATRIX"]; TAG = os.environ.get("OUT_TAG", "pidc")
    DAYS = _dc.parse_spec(os.environ.get("DAYS_SPEC", "2:4"))
    HOW = os.environ.get("PIDC_DISC", "bayesian_blocks")
    NSUB = int(os.environ.get("PIDC_CELLS", "8000"))
    A = ad.read_h5ad(MATRIX if os.path.isabs(MATRIX) else dpath(MATRIX))
    sub = A[_dc.mask(A.obs[os.environ.get("DAYCOL", "Time point")], DAYS)]
    keep = [g for g in PANEL if g in sub.var_names]
    X = sub[:, keep].X
    X = np.asarray(X.todense(), float) if sp.issparse(X) else np.asarray(X, float)
    if len(X) > NSUB:
        X = X[np.random.default_rng(0).choice(len(X), NSUB, replace=False)]
        log(f"  subsampled to {NSUB} cells (Bayesian blocks is O(n^2) per gene)")
    log(f"{X.shape[0]:,} cells x {len(keep)} genes | {len(TFS)} TFs | days {DAYS}")
    C = pidc_scores(X, keep, HOW)
    idx = {g: i for i, g in enumerate(keep)}
    rows = [dict(TF=t, target=g, importance=float(C[idx[t], idx[g]]))
            for t in TFS for g in keep if g != t]
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    _p = os.path.join(ROOT, "exports/networks", f"pidc_{TAG}.csv"); df.to_csv(_p, index=False)
    _prov.write(_p, MATRIX, DAYS, keep if "keep" in dir() else PANEL, TFS, "PIDC")
    log(f"PIDC: {len(df):,} ranked links -> pidc_{TAG}.csv")
