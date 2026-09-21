"""Conditional (partial-correlation) term for current_score, added 2026-09-18 to address the
VSC indirect-correlation confound (see handoff/2026-09-17_vsc_and_e9pos0_followup.md).

partial_corr(x, y | panel) = correlation between x and y after regressing out every other gene
in the panel -- computed via ridge-regularized precision-matrix inversion of the panel's
correlation matrix. Unlike current_score's existing terms (all bivariate/marginal statistics),
this is genuinely multivariate: it's what lets GENIE3/PIDC ignore 2-hop indirect correlation on
densely-coupled small networks like VSC, and it's the same mechanism here.

Used as a plain additive term (w=1.0, no blending with |rho|) -- the fix for the one topology
that regressed (Pluripotent, n=36 genes) is heavier ridge, not downweighting the term. A ridge
sweep (0.01 to 20) across 6 topologies (VSC, mCAD, Circadian_cycle, Pluripotent,
e9_pos0_sign_ratio, e5_pos100_density), auprc_x of |partial_corr| alone vs the plain-|rho|
baseline:

  ridge   VSC    mCAD   Circadian  Pluripotent  e9_pos0  e5_pos100
  0.01    2.362  1.209  1.334      2.146        1.623    2.139
  1.0     1.992  1.178  1.338      2.185        1.582    2.143
  8.0     1.857  1.174  1.342      2.349        1.610    2.183
  12.0    1.829  1.180  1.341      2.356        1.576    2.157
  baseline 1.698 1.178  1.370      2.369        1.603    2.211  (plain |rho|, for reference)

At low ridge, partial correlation is noisy/overconfident on Pluripotent's 36-gene precision
matrix (a genuine estimation-stability issue, not something a global constant fixes) while
VSC's signal is close to its ceiling. Heavier shrinkage (ridge~10) tames Pluripotent's estimate
back to near-baseline (2.35 vs 2.37, -0.02) while VSC keeps most of its gain (1.83-1.86 vs
1.70 baseline, +0.13-0.16); everything else moves within noise (+/-0.05). Don't scale ridge
with n_genes as a formula -- an earlier finer sweep at ridge<=1.0 showed the relationship isn't
monotonic/predictable enough to derive a clean n_genes-based rule; ridge=10 as a fixed constant
already covers the full panel-size range tested (4-36 genes) well.

DEFAULT_RIDGE=10.0 below reflects this; the term is added as-is (no weight) to current_score.
"""
import numpy as np


DEFAULT_RIDGE = 10.0


def partial_corr_matrix(X, ridge=DEFAULT_RIDGE):
    """X: cells x genes (log1p-normalized expression). Returns the |partial correlation| matrix.
    Zero-variance genes get an all-zero row/col instead of crashing the inversion."""
    std = X.std(axis=0)
    dead = std < 1e-10
    Xc = X - X.mean(axis=0, keepdims=True)
    Xc = np.divide(Xc, std[None, :] + 1e-12, out=np.zeros_like(Xc), where=~dead[None, :])
    Rmat = np.corrcoef(Xc, rowvar=False)
    Rmat = np.nan_to_num(Rmat, nan=0.0)
    np.fill_diagonal(Rmat, 1.0)
    p = Rmat.shape[0]
    P = np.linalg.pinv(Rmat + ridge * np.eye(p))
    d = np.sqrt(np.abs(np.diag(P))) + 1e-12
    pcorr = -P / np.outer(d, d)
    np.fill_diagonal(pcorr, 0.0)
    pcorr[dead, :] = 0.0
    pcorr[:, dead] = 0.0
    return pcorr


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def partial_corr_term(X, gene_names, U, ridge=DEFAULT_RIDGE):
    """X: cells x genes (log1p), columns ordered to match gene_names. U: list of (a, b) pairs
    in the SAME order the caller's score array is built in. Returns s(|partial_corr|) (ridge=10
    by default), ready to add directly into current_score's existing additive term sum --
    no separate blend weight needed, the ridge itself is what keeps it safe on large panels."""
    pcorr = partial_corr_matrix(X, ridge=ridge)
    idx = {g: i for i, g in enumerate(gene_names)}
    sc = np.array([abs(pcorr[idx[a], idx[b]]) for a, b in U])
    return s(sc)
