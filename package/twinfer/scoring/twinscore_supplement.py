"""Small statistical helpers of TwinScore_supplement (moved from paper_analysis/fatemap_pipeline/twinscore_supp_helpers.py on 2026-09-30).
REVIEW NOTE on the effective sizes: the *_pairweighted functions below are the ORIGINAL helper formulas (they treat every twin pair / cell-pair as one equally weighted unit).
They differ from the clone-weighted Kish sizes of twinfer.scoring.analytic_zscores. Measured on FM06 correlation_high (empirical sister-rho null SD 0.00939): pair-weighted twin size
-> 0.00745 (-21%), clone-weighted -> 0.01326 (+41%); on the LARRY-like cyclic_g3 Step-1 rho (true 0.01439): clone-weighted 0.01361 (-5%), the step1 pair-weighted formula 0.01633 (+13%).
Neither is exact, so they were NOT merged; the names keep the original behaviour (aliases m_eff_step1 / m_eff_twin / m_eff_cross at the bottom).
"""
import numpy as np
import pandas as pd


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def signal_share(z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < 2:
        return 0.0
    return max(0.0, 1.0 - 1.0 / z.var(ddof=1))


def clr_calibrate(M, genes):
    row_mean, row_sd = {}, {}
    for g in genes:
        vals = np.array([M.loc[g, h] for h in genes if h != g and np.isfinite(M.loc[g, h])])
        row_mean[g] = vals.mean() if len(vals) else 0.0
        row_sd[g] = vals.std(ddof=1) if len(vals) > 1 else 1.0

    def c(x, y):
        mxy = M.loc[x, y]
        if not np.isfinite(mxy):
            return np.nan
        ux = (mxy - row_mean[x]) / max(row_sd[x], 1e-12)
        uy = (mxy - row_mean[y]) / max(row_sd[y], 1e-12)
        return np.sqrt(max(0.0, ux) ** 2 + max(0.0, uy) ** 2)
    return c


def null_sd(meff):
    return 1.0 / np.sqrt(max(meff - 1.0, 1e-9))


# [2026-10-01 KNOWN ISSUE, label only, not fixed (user: this module is unlikely to be used later): the m_eff_* functions here are pair-weighted
#   (Kish without the multiplicity factor) and differ from twinfer.scoring.analytic_zscores (clone-weighted Kish). Neither is exact. Measured vs permutation SD:
#   Step-1 rho: clone-weighted -5%, pair-weighted +13%; sister rho (FM06): clone-weighted +41%, pair-weighted -21%. Rank-based scores are unaffected;
#   saved meff / z-score diagnostics from this module are only approximate. Kept as separate functions on purpose. See REVIEW_LOG.md 'm_eff'.
def m_eff_step1_pairweighted(clone_sizes):
    n = np.asarray(clone_sizes, float)
    w = 1.0 / n
    return float(w.sum() ** 2 / (w ** 2 * n).sum()) if len(n) else np.nan


def m_eff_twin_pairweighted(clone_sizes):
    n = np.asarray(clone_sizes, float)
    npairs = n * (n - 1) / 2.0
    return float(npairs.sum() ** 2 / (npairs ** 2 / npairs.clip(min=1)).sum()) if len(n) else np.nan


def m_eff_cross_pairweighted(n1, n2):
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    w = n1 * n2
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else np.nan


# names used by the fatemap pipeline (unchanged behaviour)
m_eff_step1 = m_eff_step1_pairweighted
m_eff_twin = m_eff_twin_pairweighted
m_eff_cross = m_eff_cross_pairweighted
