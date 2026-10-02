"""PIDC (Chan et al. 2017: partial information decomposition and context) -- the D term of TwinScore_supplement. Pure functions.
[Moved from paper_analysis/fatemap_pipeline/twinscore_supp_helpers.py on 2026-09-30; logic read through (Bayesian-blocks discretisation, MI, specific information, PUC, gamma-CDF
calibration of each row); no numeric change.]
"""
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


def log(msg):
    print(msg, flush=True)


def bayesian_blocks(x, p0=0.05):
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
    log(f"    PIDC discretisation: bins per gene min {nb.min()} median {int(np.median(nb))} max {nb.max()}")
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
    D = np.zeros((g, g))
    for i in range(g):
        v = PUC[i, np.arange(g) != i]
        v = v[np.isfinite(v)]
        try:
            a, loc, sc = gamma_dist.fit(v[v > 0], floc=0)
            F = lambda u, a=a, sc=sc: gamma_dist.cdf(u, a, loc=0, scale=sc)
        except Exception:
            mu, sd = float(np.mean(v)), float(np.std(v) or 1.0)
            F = lambda u, mu=mu, sd=sd: norm.cdf(u, mu, sd)
        for j in range(g):
            if i != j:
                D[i, j] += F(PUC[i, j])
    return pd.DataFrame(D, index=genes, columns=genes)
