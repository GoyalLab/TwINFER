# [copied 2026-09-30 from /gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp/twin_layers_lib.py (author/owner: yscher/laj2116), md5 9d6f406a31873c6c7350a50b4fde6866; unmodified below this header]
"""Within-cell coupling beyond what sisters predict, estimated without measurement-noise bias.

Every correlation is gzu's clone-weighted Spearman (_weighted_spearman_matrix) on the pooled twin-member rows, as calculate_gated_regulation_statistic
defines them (member weight = clone weight of the twin / 2). What is new is the input and what is done with the matrices:

 1. same-state twins: the twin table holds only sister pairs of the same annotated state (all within-clone pairs, each clone weight 1).
 2. measurement noise: UMI counts k are split into two halves by binomial thinning, k1 ~ Bin(k, 1/2), k2 = k - k1. For Poisson capture the two halves are
    independent given the cell's true expression, so a correlation between half 1 of gene x and half 2 of gene y carries no capture noise, and the
    correlation of gene x's two halves is the share of its variation that is biology (Spearman's 1904 correction, split-half).
       T = same cell, different halves          (true same-cell layer; diagonal = biological share of each gene)
       B = different cells of the clone         (inherited layer; diagonal = inherited share of each gene)
       W = T - B                                (within-cell layer beyond what sisters share)
    r_W(x,y) = W_xy / sqrt(W_xx W_yy), its standard error SE_W / sqrt(W_xx W_yy): low-count genes are not penalised, their uncertainty is carried.
 3. what sisters do not share: partial correlation of the within-cell layer given the other panel genes, from the noise-corrected r_W matrix with the
    Schafer-Strimmer analytic shrinkage (intensity = summed variance of the entries / summed squared entries; no tuning).
 4./5. ranking by probability, two samples as two measurements: three-group model over all unordered pairs for (estimate t1, estimate t2) with their own
    standard errors: no coupling / one stable coupling at both samples / unrelated couplings at the two samples. Rank by P(stable coupling).
 6. direction from time at the clone level: does the clone's level of x at t1 predict its y at t2 beyond its own y at t1, every level read through a
    different cell (cross-sibling and cross-time correlations only, so no measurement noise enters): partial lagged correlation on inherited components.
"""
import numpy as np
from scipy import optimize, stats
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package"); sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package"); sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.inference.correlation_functions import _weighted_spearman_matrix
from twinfer.scoring.analytic_zscores import kish


def _sym(A): return 0.5 * (A + A.T)


def full_depth_layers(V, ia, ib, w):
    """gzu's rho_gene_gene_2n (same-cell) and rho_cross_2n (cross-sibling, diagonal = twin intraclass correlation) for all pairs, one call."""
    p = V.shape[1]; mw = 0.5 * np.concatenate([w, w]); M = _weighted_spearman_matrix(np.vstack([np.hstack([V[ia], V[ib]]), np.hstack([V[ib], V[ia]])]), mw)
    return _sym(M[:p, :p]), _sym(M[:p, p:])


def thinned_layers(K, scale, ia, ib, w, n_thin=10, seed=0, transform=np.log1p):
    """T (same cell, across halves; diagonal = split-half biological share), B (across sisters), averaged over n_thin binomial splits of the counts."""
    rng = np.random.default_rng(seed); p = K.shape[1]; mw = 0.5 * np.concatenate([w, w]); K = np.asarray(K).astype(np.int64); T = np.zeros((p, p)); Ts = np.zeros((p, p)); B = np.zeros((p, p))
    for _ in range(n_thin):
        K1 = rng.binomial(K, 0.5); K2 = K - K1; V1 = transform(K1 * scale[:, None]); V2 = transform(K2 * scale[:, None])
        rows = np.vstack([np.hstack([V1[ia], V2[ia], V1[ib], V2[ib]]), np.hstack([V1[ib], V2[ib], V1[ia], V2[ia]])]); M = np.nan_to_num(_weighted_spearman_matrix(rows, mw))
        blk = lambda i, j: M[i * p:(i + 1) * p, j * p:(j + 1) * p]
        T += _sym(blk(0, 1)); Ts += 0.5 * (blk(0, 0) + blk(1, 1)); B += 0.25 * (_sym(blk(0, 3)) + _sym(blk(1, 2)) + _sym(blk(0, 2)) + _sym(blk(1, 3)))
    T /= n_thin; Ts /= n_thin; B /= n_thin; Tfull = 0.5 * (T + Ts); np.fill_diagonal(Tfull, np.diag(T))      # off-diagonal: all four half combinations; diagonal: across halves only
    return Tfull, B, float(kish(mw)), float(kish(w))


def corrected(T, B, m_same, m_twin):
    seT = 1.0 / np.sqrt(max(m_same - 1, 1.0)); seB = 1.0 / np.sqrt(max(m_twin - 1, 1.0)); seW = float(np.sqrt(seT ** 2 + seB ** 2)); out = {}
    for name, Mx, se in (("T", T, seT), ("B", B, seB), ("W", T - B, seW)):
        d = np.maximum(np.diag(Mx), 2 * se); N = np.sqrt(np.outer(d, d)); R = Mx / N; np.fill_diagonal(R, 1.0); SE = se / N; np.fill_diagonal(SE, 0.0)
        out[name] = dict(r=R, se=SE, diag=np.diag(Mx).copy(), reliable=np.diag(Mx) > 2 * se, se0=se)
    return out


def shrunk_partial(R, SE):
    """Schafer-Strimmer: R* = (1-a) R + a I with a = sum Var(r_ij) / sum r_ij^2 over i != j; partial correlations from the inverse."""
    p = len(R); off = ~np.eye(p, dtype=bool); Rc = np.clip(R, -1, 1); a = float(np.clip((SE[off] ** 2).sum() / max((Rc[off] ** 2).sum(), 1e-12), 0.0, 1.0)); Rs = (1 - a) * Rc + a * np.eye(p)
    w_, V_ = np.linalg.eigh(_sym(Rs)); Rs = (V_ * np.maximum(w_, 1e-3)) @ V_.T; d = np.sqrt(np.diag(Rs)); Rs = Rs / np.outer(d, d); P = np.linalg.inv(Rs); dd = np.sqrt(np.diag(P)); pc = -P / np.outer(dd, dd); np.fill_diagonal(pc, 1.0)
    return pc, (1 - a) * SE, a


def three_group(t1, s1, t2, s2, n_start=6, seed=0):
    """(t1, t2) ~ pi0 N(0, diag(s^2 + tau0^2)) + pi1 N(0, diag(s^2) + tau1^2 11') + pi2 N(0, diag(s^2 + tau2^2)), tau2 > tau0. Returns log posterior odds of the stable group."""
    t1, s1, t2, s2 = (np.asarray(v, float) for v in (t1, s1, t2, s2)); ok = np.isfinite(t1) & np.isfinite(t2) & np.isfinite(s1) & np.isfinite(s2)
    def comp(par):
        l0, l1, l2, a0, a1, a2 = par; lp = np.array([l0, l1, l2]); lp = lp - np.logaddexp.reduce(lp); tau0, tau1 = np.exp(a0), np.exp(a1); tau2 = tau0 + np.exp(a2)
        v1, v2 = s1 ** 2 + tau0 ** 2, s2 ** 2 + tau0 ** 2; L0 = lp[0] + stats.norm.logpdf(t1, 0, np.sqrt(v1)) + stats.norm.logpdf(t2, 0, np.sqrt(v2))
        v1, v2 = s1 ** 2 + tau2 ** 2, s2 ** 2 + tau2 ** 2; L2 = lp[2] + stats.norm.logpdf(t1, 0, np.sqrt(v1)) + stats.norm.logpdf(t2, 0, np.sqrt(v2))
        a, b, c = s1 ** 2 + tau1 ** 2, tau1 ** 2, s2 ** 2 + tau1 ** 2; det = a * c - b * b; q = (c * t1 ** 2 - 2 * b * t1 * t2 + a * t2 ** 2) / det; L1 = lp[1] - np.log(2 * np.pi) - 0.5 * np.log(det) - 0.5 * q
        return L0, L1, L2
    def nll(par): L0, L1, L2 = comp(par); return -np.logaddexp.reduce([L0[ok], L1[ok], L2[ok]], axis=0).sum()
    rng = np.random.default_rng(seed); sc = float(np.nanstd(np.r_[t1[ok], t2[ok]])) or 1.0; best = None
    for _ in range(n_start):
        x0 = np.r_[rng.normal(0, 1, 3), np.log(sc * rng.uniform(0.02, 0.3)), np.log(sc * rng.uniform(0.3, 1.5)), np.log(sc * rng.uniform(0.3, 1.5))]
        r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=[(-8, 8)] * 3 + [(np.log(sc) - 9, np.log(sc) + 3)] * 3)
        if best is None or r.fun < best.fun: best = r
    L0, L1, L2 = comp(best.x); lo = L1 - np.logaddexp(L0, L2); lp = best.x[:3] - np.logaddexp.reduce(best.x[:3])
    return lo, dict(pi=np.exp(lp).round(3).tolist(), tau0=float(np.exp(best.x[3])), tau1=float(np.exp(best.x[4])), tau2=float(np.exp(best.x[3]) + np.exp(best.x[5])), nll=float(best.fun))


def lagged_direction(C12, B1, B2diag, se_cross, se_b1, se_b2):
    """g(x->y) = partial correlation of x(t1) with y(t2) given y(t1), on inherited components: a = C12_xy / sqrt(B1_xx B2_yy), b = B1_xy / sqrt(B1_xx B1_yy),
    c = C12_yy / sqrt(B1_yy B2_yy); g = (a - b c) / sqrt((1 - b^2)(1 - c^2)). z = (|g_xy| - |g_yx|) / SE."""
    d1 = np.maximum(np.diag(B1), 2 * se_b1); d2 = np.maximum(B2diag, 2 * se_b2); a = C12 / np.sqrt(np.outer(d1, d2)); b = np.clip(B1 / np.sqrt(np.outer(d1, d1)), -0.95, 0.95); c = np.clip(np.diag(C12) / np.sqrt(d1 * d2), -0.95, 0.95)
    den = np.sqrt((1 - b ** 2) * (1 - c[None, :] ** 2)); g = (a - b * c[None, :]) / den; se = se_cross / np.sqrt(np.outer(d1, d2)) / den; z = (np.abs(g) - np.abs(g.T)) / np.sqrt(se ** 2 + se.T ** 2); np.fill_diagonal(z, 0.0)
    return g, se, z


def sign_posterior(z):
    """z ~ (1-w) N(0,1) + w N(0, 1 + d^2); P(true asymmetry > 0 | z) = (1-p2)/2 + p2 Phi(z sqrt(d^2/(1+d^2)))."""
    z = np.asarray(z, float); zz = z[np.isfinite(z)]
    def nll(par): w = 1 / (1 + np.exp(-par[0])); d2 = np.exp(par[1]); return -np.logaddexp(np.log1p(-w) + stats.norm.logpdf(zz), np.log(w) + stats.norm.logpdf(zz, 0, np.sqrt(1 + d2))).sum()
    r = min((optimize.minimize(nll, x0, method="L-BFGS-B", bounds=[(-10, 10), (-8, 6)]) for x0 in ([0.0, 0.0], [-2.0, 1.0], [2.0, -1.0])), key=lambda r_: r_.fun)
    w = 1 / (1 + np.exp(-r.x[0])); d2 = np.exp(r.x[1]); l2 = np.log(w) + stats.norm.logpdf(z, 0, np.sqrt(1 + d2)); l1 = np.log1p(-w) + stats.norm.logpdf(z); p2 = np.exp(l2 - np.logaddexp(l1, l2))
    return (1 - p2) * 0.5 + p2 * stats.norm.cdf(z * np.sqrt(d2 / (1 + d2))), dict(w=float(w), d2=float(d2))
