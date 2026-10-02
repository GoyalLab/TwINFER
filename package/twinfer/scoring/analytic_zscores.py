"""[Moved from paper_analysis/larry_hematopoiesis_validation/analytic_zscores.py on 2026-09-30 after review: math checked (half-normal fold, Kish sizes, z_gamma variance) and
the column set equals infer_with_twinfer's twin_score_inputs; calibrated SD defaults are LARRY days 2 & 4 only.]
Shuffle-free ("analytic") approximation of every z-score infer_with_twinfer returns.

Motivation
----------
Each TwINFER z-score is a permutation test of a clone-weighted Spearman correlation (or a linear
combination of them). A permutation statistic that is *linear in the permutation*
(``sum_i w_i a_i b_{pi(i)}``) has a permutation mean and variance available in closed form
(Pitman 1937 / Hoeffding 1952): mean 0, variance ~ 1/(m_eff - 1). So the 5000-shuffle null can be
replaced by a Gaussian whose two moments are computed directly, with three correction terms:

  1. m_eff = the clone-weighted **Kish effective sample size**, not the raw cell / clone / twin
     count. For weights w (one per exchangeable unit):  m_eff = (sum w)^2 / sum(w^2).
       - step1 (gene-gene rho): unit = clone, cell weight 1/n_c  ->  m = n_clone^2 / sum(1/n_c)
       - twin-delta stats (unit="clone"): each clone weight 1 split over its C(n_c,2) twins  ->
         m = n_twinclone^2 / sum(1 / C(n_c,2))
       - cross-time (direction): clone weight 1 split over its n2*n4 cross pairs

  2. **calibrated null SD**. The pure 1/sqrt(m_eff-1) is 2-9% off depending on the null's
     construction (clone-matching vs fixed-Delta re-pairing vs cell-time-label shuffle). The null
     SD is a property of the dataset's clone structure, *not* the gene set (CV < 4% across gene
     sets and pairs), so it is measured once from any completed 5000-shuffle run and reused.
     Defaults below are for LARRY days 2 & 4 (larry_qc_counts, ~32.4k cells).

  3. **null centre**. Zero for the "destroy all X-Y association" nulls (step1, z_div, z_d_div,
     z_rho_cross, z_rho_change). NON-zero for the "fresh unrelated pair" nulls (z_het, z_d_het):
     the centre is E[rho_Delta of random pairs], estimated from ~20-60 cheap re-pairings (two
     orders of magnitude fewer than the full null) -- ``random_delta_reference_matrix`` in the
     infer.py output already stores one such draw.

  For the |.|-statistics (z_abs_rho_t1/t2/change, z_gamma) the null draw X ~ N(0, s) is folded:
  |X| is half-normal with mean s*sqrt(2/pi) and SD s*sqrt(1 - 2/pi). z_gamma = |A| - |B| for two
  ~independent half-normals -> SD s*sqrt(2*(1 - 2/pi)).

Validation (plot_analytic_vs_shuffle.py): on the 6-scenario drift sims the analytic z matches the
5000-shuffle z to a median |dz| of 0.01-0.13 for the 0-centred statistics and ~0.03 for z_het
(given its centre); the permutation nulls are Gaussian there (|skew| ~ 0.04, empirical 99th
percentile within ~1% of 2.326 sigma). On real LARRY data the agreement is the same.

Not fully analytic: z_reg_gated's null has Y-sibling-block structure, so its analytic SD is only
approximate (use SD_HET_T1 as a proxy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

S2PI = np.sqrt(2.0 / np.pi)          # E|X| / sigma   for X ~ N(0, sigma)
VHALF = 1.0 - 2.0 / np.pi            # Var|X| / sigma^2

# ----------------------------------------------------------------------------------------------
# Calibrated permutation-null SDs -- LARRY (larry_qc_counts, days 2 & 4, unit="clone",
# use_clone=True). Measured from the 5000-shuffle runs (correlation_high / detection_mid; median
# over pairs, CV < 4%). Structure-only: independent of the gene set.
# ----------------------------------------------------------------------------------------------
SD_STEP1_T1 = 0.0172     # step1 co-expression rho, day 2   (== rho_t1_for_twinscore null)
SD_STEP1_T2 = 0.0143     # step1 co-expression rho, day 4   (== rho_t2_for_twinscore null)
SD_DIV_T1 = 0.0256       # z_div: fixed-Delta re-pairing, day 2
SD_DIV_T2 = 0.0154       # z_div: fixed-Delta re-pairing, day 4
SD_HET_T1 = 0.0250       # z_het: fresh random-pair rho_Delta, day 2
SD_D = 0.0300            # z_d_het / z_d_div: d = rho_Delta(t2) - rho_Delta(t1)
SD_CHANGE = 0.0205       # z_rho_change / z_abs_rho_change: cell-time-label shuffle of Delta-rho
SD_CROSS = 0.0237        # z_rho_cross / z_gamma: paired cross-clone permutation of t2 rows

DEFAULT_SD = dict(step1_t1=SD_STEP1_T1, step1_t2=SD_STEP1_T2, div_t1=SD_DIV_T1, div_t2=SD_DIV_T2,
                  het_t1=SD_HET_T1, d=SD_D, change=SD_CHANGE, cross=SD_CROSS)


# ----------------------------------------------------------------------------------------------
# Effective sample sizes from the clone-size distribution (use when no calibrated SD is
# available -- then sd = 1 / sqrt(m_eff - 1)).
# ----------------------------------------------------------------------------------------------
def kish(weights) -> float:
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def m_eff_step1(clone_sizes) -> float:
    """Gene-gene rho, use_clone=True: cell weight 1/n_c, clone weight 1."""
    n = np.asarray(clone_sizes, float)
    return float(len(n) ** 2 / (1.0 / n).sum())


def m_eff_twin(clone_sizes) -> float:
    """Twin-delta rho, unit='clone': clone weight 1 split over its C(n_c, 2) enumerated twins."""
    n = np.asarray(clone_sizes, float)
    n = n[n >= 2]
    c = n * (n - 1) / 2.0
    return float(len(n) ** 2 / (1.0 / c).sum())


def m_eff_cross(n2, n4) -> float:
    """Cross-time rho: clone weight 1 split over its n2 * n4 cross-time pairs."""
    n2, n4 = np.asarray(n2, float), np.asarray(n4, float)
    return float(len(n2) ** 2 / (1.0 / (n2 * n4)).sum())


def m_eff_d(m_twin_t1, m_twin_t2) -> float:
    """d = rho_Delta(t2) - rho_Delta(t1), disjoint clone sets -> Var adds."""
    return 1.0 / (1.0 / m_twin_t1 + 1.0 / m_twin_t2)


def m_effs_from_table(df, t1=2, t2=4, clone_col="clone_id", time_col="time_step") -> dict:
    out = {}
    for tp, tag in ((t1, "t1"), (t2, "t2")):
        g = df[df[time_col] == tp].groupby(clone_col).size()
        out[f"step1_{tag}"] = m_eff_step1(g.to_numpy())
        out[f"twin_{tag}"] = m_eff_twin(g.to_numpy())
    d2 = set(df[df[time_col] == t1][clone_col]); d4 = set(df[df[time_col] == t2][clone_col])
    both = sorted(d2 & d4)
    g2 = df[df[time_col] == t1].groupby(clone_col).size()
    g4 = df[df[time_col] == t2].groupby(clone_col).size()
    out["cross"] = m_eff_cross([g2[c] for c in both], [g4[c] for c in both])
    out["d"] = m_eff_d(out["twin_t1"], out["twin_t2"])
    return out


# ----------------------------------------------------------------------------------------------
# The z-scores.
# ----------------------------------------------------------------------------------------------
def z_signed(rho, sd, centre=0.0):
    """z for a signed statistic with a ~N(centre, sd) permutation null. step1, z_div, z_d_div,
    z_rho_cross, z_rho_change, and z_het / z_d_het (with a non-zero centre)."""
    return (np.asarray(rho, float) - centre) / sd


def z_abs(rho, sd):
    """z for a |.|-statistic: null draw X ~ N(0, sd), folded to the half-normal |X|.
    z = (|rho| - sd*sqrt(2/pi)) / (sd*sqrt(1 - 2/pi)).  z_abs_rho_t1/t2/change."""
    return (np.abs(np.asarray(rho, float)) - sd * S2PI) / (sd * np.sqrt(VHALF))


def z_gamma(gamma, sd_cross):
    """gamma = |rho_cross(x->y)| - |rho_cross(y->x)|. Null |A| - |B|, A,B ~ N(0, sd_cross)
    ~independent -> Var = 2 * sd_cross^2 * (1 - 2/pi)."""
    return np.asarray(gamma, float) / (sd_cross * np.sqrt(2.0 * VHALF))


def z_reg_gated(rho_same, rho_cross_2n, z_het, sd_reg=SD_HET_T1):
    """rho_reg(lambda) = rho_same - lambda * rho_cross,  lambda = min(1, |z_het| / 2.33).
    Analytic SD is approximate (Y-sibling-block null); SD_HET_T1 is a proxy."""
    lam = np.minimum(1.0, np.abs(np.asarray(z_het, float)) / 2.33)
    return (np.asarray(rho_same, float) - lam * np.asarray(rho_cross_2n, float)) / sd_reg


def analytic_twin_score_inputs(rho, SD=DEFAULT_SD, het_centre_t1=None, het_centre_d=None):
    """Assemble the analytic twin_score_inputs table (columns calculate_twin_score expects).

    `rho` : dict of gene x gene DataFrames --
        rho_t1, rho_t2                gene-gene Spearman at t1, t2 (calculate_pairwise_gene_gene_correlation_matrix)
        rho_delta_t1, rho_delta_t2    twin rho_Delta at t1, t2      (calculate_twin_random_correlations[0])
        rho_delta_random_t1/_t2       random-pair rho_Delta reference (mean of ~20-60 draws)
        rho_cross                    directed cross-time Spearman   (get_cross_correlations)
    """
    m1 = rho["rho_t1"]
    genes = list(m1.index)
    r1m, r2m = rho["rho_t1"], rho["rho_t2"]
    d1m, d2m = rho["rho_delta_t1"], rho["rho_delta_t2"]
    rr1m = het_centre_t1 if het_centre_t1 is not None else rho["rho_delta_random_t1"]
    rr2m = rho["rho_delta_random_t2"]
    xcm = rho["rho_cross"]
    rows = []
    for a in genes:
        for b in genes:
            if a == b:
                continue
            rt1, rt2 = float(r1m.loc[a, b]), float(r2m.loc[a, b])
            rd1, rd2 = float(d1m.loc[a, b]), float(d2m.loc[a, b])
            rr1, rr2 = float(rr1m.loc[a, b]), float(rr2m.loc[a, b])
            rxy, ryx = float(xcm.loc[a, b]), float(xcm.loc[b, a])
            rchg, d = rt2 - rt1, rd2 - rd1
            dc = float(het_centre_d.loc[a, b]) if het_centre_d is not None else (rr2 - rr1)
            gam = abs(rxy) - abs(ryx)
            rows.append(dict(
                gene_1=a, gene_2=b,
                rho_t1=rt1, rho_t2=rt2, rho_delta_t2=rd2, rho_change=rchg,
                z_abs_rho_t1=float(z_abs(rt1, SD["step1_t1"])),
                z_abs_rho_t2=float(z_abs(rt2, SD["step1_t2"])),
                z_abs_rho_change=float(z_abs(rchg, SD["change"])),
                z_rho_change=float(z_signed(rchg, SD["change"])),
                z_div=float(z_signed(rd1, SD["div_t1"])),
                z_het=float(z_signed(rd1, SD["het_t1"], centre=rr1)),
                z_d_het=float(z_signed(d, SD["d"], centre=dc)),
                z_d=float(z_signed(d, SD["d"], centre=dc)),
                rescued_from_no_regulation=False,
                rho_cross_xy=rxy, rho_cross_yx=ryx,
                gamma=gam, z_gamma=float(z_gamma(gam, SD["cross"])),
                is_final_directed_edge=False,
            ))
    return pd.DataFrame(rows)


__all__ = ["S2PI", "VHALF", "DEFAULT_SD", "kish", "m_eff_step1", "m_eff_twin", "m_eff_cross",
           "m_eff_d", "m_effs_from_table", "z_signed", "z_abs", "z_gamma", "z_reg_gated",
           "analytic_twin_score_inputs"]
