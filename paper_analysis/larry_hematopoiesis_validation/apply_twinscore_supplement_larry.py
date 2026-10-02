"""Compute every term of TwinScore_supplement.pdf (the 2026-09-17 TwinScore supplement to
TwINFER) on real LARRY hematopoiesis data, for yscher's correlation_high and correlation_mid
gene panels, day2 -> day4 (t1=2, t2=4; day 4 is the larger sample, 8,959 vs 4,574 cells),
unfiltered twin definition (clone_id = larry_clone_singletcode, following
preprocessing/build_twinfer_inputs_per_geneset.py).

Per 2026-09-18 user instruction: every term is recomputed FRESH and directly from the raw
twin-pair cell data / raw UMI counts. Nothing here reads any pre-computed twin_score_inputs.csv
/ z_scores_by_step.json / networks_yscher/*.csv from the older manuscript-stage pipeline for the
TwinScore terms themselves. The only things reused are (a) the package's already-validated
weighted-Spearman / twin-pair-construction primitives in correlation_functions.py, which ARE the
literal mechanics the supplement describes (not a shortcut around them), and (b) two calibrated
constants -- SD["het_t1"] for z_het and SD["cross"] for z^dagger -- because the supplement itself
states z_het IS the manuscript's existing stage-II statistic and z^dagger IS the manuscript's
existing cross-correlation, so recomputing them from different first principles would make them a
*different* quantity, not the same one measured twice. The competitor comparison bar reuses the
existing networks_yscher/*_allgenes.csv (best of 5 methods, pooled day2+4 cells) since this repo
has no separate per-timepoint competitor runs (the supplement's own bar is 5 methods x 2 samples).

Two terms are genuinely ambiguous in the PDF text and are resolved with a disclosed convention
(search "DISCLOSED" below): the exact T/B split for the optional Wz term, and the numeric cutoffs
used only for the Section-5 scenario labels (a labeling convenience, not part of the score).

Outputs, per gene set, into this script's directory:
  {gs}_gene_terms.csv    one row per gene: h(t1), h(t2), phi, scenario class inputs
  {gs}_pair_terms.csv    one row per ordered pair: every quantity in section 3-4, TwinScore, PAIR,
                         scenario label
  twinscore_supplement_larry_summary.csv   Section-6-style evaluation table (both gene sets)
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from scipy.stats import gamma as gamma_dist
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, split_twins, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
    get_clone_weights, get_unit_weights, weighted_spearman,
)

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f"{HERE}/resources"
R = f"{RES_HERE}/resources"
SOURCE = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed, z_abs, m_eff_step1, m_eff_twin, m_eff_cross

GENE_SETS = os.environ.get("GENE_SETS", "correlation_high,correlation_mid").split(",")
T1, T2 = 2, 4
SEED = 0
N_RAND_HET = 40
R0 = 0.69
Z_DIRECTION_GATE = 2.576   # section 3.3 direction-term gate
Z_ONE_SIDED = 2.326        # lambda cap and R-present cutoff, both "2.33"/"2.326" in the PDF


def log(msg):
    print(msg, flush=True)


# =================================================================================================
# small shared helpers
# =================================================================================================
def s(v):
    """s(.): z-standardise over all ordered pairs of the panel (section 2)."""
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def signal_share(z):
    """Signal share of a z-valued pair statistic (section 2): max(0, 1 - 1/var(z))."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < 2:
        return 0.0
    return max(0.0, 1.0 - 1.0 / z.var(ddof=1))


def clr_calibrate(M, genes):
    """CLR calibration of a symmetric pair matrix (section 2)."""
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


# =================================================================================================
# data loading
#
# 2026-09-18: switched from TwINFER's own recreation of yscher's panels (resources/
# gene_sets_yscher.json, built by pick_gene_sets.ipynb) to yscher's ACTUAL panels as she ran them
# -- pulled directly from her own project's per-gene-set TwinScore JSONs
# (/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/twinscore_script/
#  larry_log1pPF_stable24_det5_p4a01_{corrhigh,corrmid}_24.json's "genes" field). Verified against
# her own helpers/twinscore_table.py: running it reproduces the PDF's exact "LARRY day2-day4"
# high-co-expression row (107 calls, best-competitor PIDC 2.08x precision multiplier, both exact
# matches), and her gene list is NOT identical to gene_sets_yscher.json's (47 vs 44 genes for
# "high"; 76 vs 68 for "mid") -- that gap fully explains the earlier mismatch.
# =================================================================================================
YCOPY = f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports"  # [2026-10-01 added: copy of yscher exports (twinscore_script, twinscore_gated, panels/tf_target_panel, _probe_tmp gene_flags)]
# [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] YSCHER_ROOT = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/twinscore_script"
YSCHER_ROOT = f"{YCOPY}/twinscore_script"
YSCHER_STEM = {"correlation_high": "larry_log1pPF_stable24_det5_p4a01_corrhigh_24",
               "correlation_mid": "larry_log1pPF_stable24_det5_p4a01_corrmid_24",
               "correlation_low": "larry_log1pPF_stable24_det5_p4a01_corrlow_24",
               # "variability_*" -- dispersion-quantile bands, independent of any TF-target
               # correlation ranking (unlike the correlation_* panels above). yscher calls these
               # "spread" (hvg_q<band>) in her own file naming.
               "variability_high": "larry_log1pPF_stable24_det5_p4a01_hvg_q67100_24",
               "variability_mid": "larry_log1pPF_stable24_det5_p4a01_hvg_q3367_24",
               "variability_low": "larry_log1pPF_stable24_det5_p4a01_hvg_q0033_24"}


def yscher_genes(gs):
    d = json.load(open(f"{YSCHER_ROOT}/{YSCHER_STEM[gs]}.json"))
    return sorted(d["genes"])


def yscher_real_permutation_terms(gs, genes):
    """S, z_het, and the directional cross-sample terms, taken VERBATIM from yscher's own real
    -permutation run (helpers/twinscore_script_only.py) rather than recomputed with an analytic
    null-sd shortcut. Per 2026-09-18 user instruction: match her actual methodology -- she calls
    the package's own functions directly (check_gene_gene_correlation_threshold,
    differentiate_single_state_reg_and_multiple_states, identify_actual_directed_edges) with the
    module's adaptive real-permutation draw count (not a fixed calibrated SD constant), and she
    already ran this on exactly these two panels. Only her genes' ORDER may differ from `genes`
    (both are the same set, sorted identically here since both are sorted()).

    Returns matrices S (=rho_t2), S_t1 (=rho_t1), zhet, rho_dagger (off-diagonal only, diagonal
    left NaN -- persistence's self-correlation is a supplement-only quantity she never computes,
    filled in separately from our own get_cross_correlations call), z_dagger (same), and the two
    per-pair real Step-I z-scores z_rho_t1/z_rho_t2 (for the scenario-I significance check).
    """
    d = json.load(open(f"{YSCHER_ROOT}/{YSCHER_STEM[gs]}.json"))
    assert sorted(d["genes"]) == genes, "gene list mismatch vs yscher_genes()"
    pairs = [tuple(p) for p in d["pairs"]]
    mats = {k: pd.DataFrame(np.nan, index=genes, columns=genes)
            for k in ("S", "S_t1", "zhet", "rho_dagger", "z_dagger", "z_rho_t1", "z_rho_t2")}
    for i, (a, b) in enumerate(pairs):
        mats["S"].loc[a, b] = mats["S"].loc[b, a] = d["rho_t2"][i]
        mats["S_t1"].loc[a, b] = mats["S_t1"].loc[b, a] = d["rho_t1"][i]
        mats["zhet"].loc[a, b] = mats["zhet"].loc[b, a] = d["z_het"][i]
        mats["rho_dagger"].loc[a, b] = d["fwd"][i]
        mats["rho_dagger"].loc[b, a] = d["rev"][i]
        mats["z_dagger"].loc[a, b] = d["z_fwd"][i]
        mats["z_dagger"].loc[b, a] = d["z_rev"][i]
        mats["z_rho_t1"].loc[a, b] = mats["z_rho_t1"].loc[b, a] = d["z_rho_t1"][i]
        mats["z_rho_t2"].loc[a, b] = mats["z_rho_t2"].loc[b, a] = d["z_rho_t2"][i]
    return mats


def load_raw(gs):
    """Build the canonical clone_id/cell_id/time_step/{gene}_mRNA table directly from the raw LARRY
    counts (larry_qc_counts.mtx), for yscher's own gene list -- log1p(CP10k), clone_id =
    larry_clone_singletcode (unfiltered), day parsed from library, exactly matching
    preprocessing/build_twinfer_inputs_per_geneset.py's convention."""
    import scipy.io as sio
    genes = yscher_genes(gs)
    X = sio.mmread(f"{SOURCE}/larry_qc_counts.mtx").tocsr()
    genes_full = np.array(open(f"{SOURCE}/genes.txt").read().split())
    obs = pd.read_csv(f"{SOURCE}/obs_metadata.csv", index_col=0)
    day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False).astype(int)
    gidx = {g: i for i, g in enumerate(genes_full)}
    col = [gidx[g] for g in genes]
    cell_total = np.asarray(X.sum(axis=1)).ravel().astype(float)
    mat = np.log1p(X[:, col].toarray().astype(float) / cell_total[:, None] * 1e4)
    df = pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in genes])
    df.insert(0, "time_step", day.to_numpy())
    df.insert(0, "cell_id", obs.index.to_numpy())
    df.insert(0, "clone_id", obs["larry_clone_singletcode"].to_numpy())
    t1_raw = df[df.time_step == T1].reset_index(drop=True)
    t2_raw = df[df.time_step == T2].reset_index(drop=True)
    return t1_raw, t2_raw, genes


# =================================================================================================
# S(x,y), C(x,y) / heritability, zhet
# =================================================================================================
def same_cell_matrix(raw, genes):
    """S(x,y): same-cell correlation of x and y, clone weight 1/n_c per cell (section 3.2)."""
    return calculate_pairwise_gene_gene_correlation_matrix(raw, genes, use_clone=True)


def sister_matrix_from_frame(raw, genes):
    """C(x,y): sister-sister cross-gene correlation, x in one twin against y in the other, BOTH
    orderings pooled (section 3.2). Diagonal C(g,g) is heritability h_g at this sample.

    Uses every within-clone twin pair (assign_twin_id already enumerates all C(n_c,2) unordered
    pairs per clone, not one arbitrarily-chosen pair), clone weight 1 split evenly over its twins.
    A twin {i,j} contributes two pooled observations -- (x_i,y_j) and (x_j,y_i) -- each at half the
    twin's weight, since the twin's two cells carry no intrinsic order.
    """
    tw = assign_twin_id(raw)
    twin_0, twin_1 = split_twins(tw)
    w = get_unit_weights(twin_0, unit="clone")       # 1 / C(n_c,2) per twin
    w2 = np.concatenate([w, w]) / 2.0
    C = pd.DataFrame(np.nan, index=genes, columns=genes)
    vals0 = {g: twin_0[f"{g}_mRNA"].to_numpy() for g in genes}
    vals1 = {g: twin_1[f"{g}_mRNA"].to_numpy() for g in genes}
    for i, gx in enumerate(genes):
        for gy in genes[i:]:
            px = np.concatenate([vals0[gx], vals1[gx]])
            py = np.concatenate([vals1[gy], vals0[gy]])
            r = weighted_spearman(px, py, w2)
            C.loc[gx, gy] = C.loc[gy, gx] = r
    clone_sizes = raw.groupby("clone_id").size().to_numpy()
    meff = m_eff_twin(clone_sizes)
    return C, meff


def het_stats(raw, genes, n_rand=N_RAND_HET, seed=SEED):
    """z_het = the manuscript's stage-II statistic, reused verbatim (the supplement states it IS
    this quantity): twin-difference rho minus its random-pairing centre, in SD["het_t1"] units."""
    tw = assign_twin_id(raw)
    rho_delta_obs, _ = calculate_twin_random_correlations(raw, tw, genes, random_state=seed, unit="clone")
    draws = [calculate_twin_random_correlations(raw, tw, genes, random_state=seed + 100 + i, unit="clone")[1].to_numpy()
             for i in range(n_rand)]
    rho_delta_random = pd.DataFrame(np.mean(draws, axis=0), index=genes, columns=genes)
    return rho_delta_obs, rho_delta_random


# =================================================================================================
# cross-sample twin correlation: persistence numerator + z^dagger inputs
# =================================================================================================
def cross_matrix(t1_raw, t2_raw, genes):
    """rho^dagger(x,y): x in the t1 cell against y in its t2 clone-mate, directed, all clone-t1 x
    clone-t2 combinations (section 3.3 / 3.1). Diagonal = rho^dagger_gg, the persistence numerator."""
    at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
    ordered = [(a, b) for a in genes for b in genes if a != b]
    xc = get_cross_correlations(at1, at2, gene_pairs=ordered, unit="clone")
    both = set(t1_raw.clone_id) & set(t2_raw.clone_id)
    n2 = t1_raw[t1_raw.clone_id.isin(both)].groupby("clone_id").size()
    n4 = t2_raw[t2_raw.clone_id.isin(both)].groupby("clone_id").size()
    both = sorted(both)
    meff = m_eff_cross(n2.loc[both].to_numpy(), n4.loc[both].to_numpy())
    return xc, meff


# =================================================================================================
# persistence phi_g and its reliability w (section 3.1)
# =================================================================================================
def bootstrap_phi_noise_floor(t1_raw, t2_raw, genes, thr1, thr2, n_boot=20, seed=SEED):
    """Real empirical (clone-level cluster bootstrap) noise variance of phi_g, replacing the
    analytic delta-method proxy.

    2026-09-19 fix #1: the delta-method formula (var(phi) ~ phi^2 * [var(rho_dagger)/rho_dagger^2 +
    var(h1)/(4h1^2) + var(h2)/(4h2^2)]) blows up whenever h1/h2 are small and noisy -- exactly the
    case on every panel except the two correlation-selected "high" ones (median h ~0.2 there vs
    ~0.03-0.05 elsewhere) -- so it overshot the true cross-gene variance of phi and collapsed
    w = max(0, 1-floor/var) to exactly 0.00 on correlation_mid, correlation_low, variability_mid,
    variability_low, and the top-100-detected-TF/random-TF checks. This is the same failure mode
    already fixed for Wz's v-gate: resample clones (the true exchangeable unit) directly, with
    replacement, rebuild h1/h2/rho_dagger from each resampled draw, and use the REAL empirical
    variance of phi_g across draws as its noise floor.

    2026-09-19 fix #2: the first version of this bootstrap only required h1,h2 > 0 within each
    replicate (not the point estimate's own significance gate h > 2*null_sd), so on low-heritability
    panels a handful of replicates would push h1 or h2 to a tiny-but-positive value; dividing by
    sqrt(h1*h2) near zero produced enormous spurious phi outliers (checked directly on
    correlation_mid: bootstrap noise variance ranged 0.0009 to 146, median 0.72, against a true
    observed cross-gene variance of only 0.028 -- a 26x-inflated floor). Applying the SAME
    thr1/thr2 gate inside each replicate that the point estimate uses keeps the noise floor
    measuring real sampling variability of phi itself, not of an unstable near-zero-denominator
    ratio the point estimate would never have accepted either.
    """
    clone_idx_t1 = t1_raw.groupby("clone_id").indices
    clone_idx_t2 = t2_raw.groupby("clone_id").indices
    universe = sorted(set(t1_raw.clone_id) | set(t2_raw.clone_id))
    rng = np.random.default_rng(seed)
    phi_draws = {g: [] for g in genes}
    for b in range(n_boot):
        drawn = rng.choice(universe, size=len(universe), replace=True)
        t1_idx, t1_cid, t2_idx, t2_cid = [], [], [], []
        for i, c in enumerate(drawn):
            r1 = clone_idx_t1.get(c)
            if r1 is not None and len(r1):
                t1_idx.append(r1); t1_cid.append(np.full(len(r1), i))
            r2 = clone_idx_t2.get(c)
            if r2 is not None and len(r2):
                t2_idx.append(r2); t2_cid.append(np.full(len(r2), i))
        if not t1_idx or not t2_idx:
            continue
        t1_b = t1_raw.iloc[np.concatenate(t1_idx)].reset_index(drop=True).copy()
        t1_b["clone_id"] = np.concatenate(t1_cid)
        t2_b = t2_raw.iloc[np.concatenate(t2_idx)].reset_index(drop=True).copy()
        t2_b["clone_id"] = np.concatenate(t2_cid)
        try:
            C1b, _ = sister_matrix_from_frame(t1_b, genes)
            C2b, _ = sister_matrix_from_frame(t2_b, genes)
            xcb, _ = cross_matrix(t1_b, t2_b, genes)
        except ValueError:
            continue
        for g in genes:
            h1g, h2g, rdg = C1b.loc[g, g], C2b.loc[g, g], xcb.loc[g, g]
            ok = np.isfinite(h1g) and np.isfinite(h2g) and h1g > thr1 and h2g > thr2
            denom = np.sqrt(h1g * h2g) if (ok and h1g > 0 and h2g > 0) else np.nan
            phib = rdg / denom if (np.isfinite(denom) and denom > 0) else np.nan
            if np.isfinite(phib):
                phi_draws[g].append(phib)
    noise_var = {}
    for g in genes:
        vals = np.array(phi_draws[g])
        if len(vals) >= 4:
            noise_var[g] = float(np.var(vals, ddof=1))
    return noise_var


def persistence(h1, h2, rho_dagger, genes, meff_twin_t1, meff_twin_t2, meff_cross_, t1_raw, t2_raw):
    var_h1, var_h2 = null_sd(meff_twin_t1) ** 2, null_sd(meff_twin_t2) ** 2
    thr1, thr2 = 2.0 * np.sqrt(var_h1), 2.0 * np.sqrt(var_h2)
    phi_raw = {}
    for g in genes:
        h1g, h2g, rdg = h1[g], h2[g], rho_dagger[g]
        ok = np.isfinite(h1g) and np.isfinite(h2g) and h1g > thr1 and h2g > thr2
        denom = np.sqrt(h1g * h2g) if ok and h1g > 0 and h2g > 0 else np.nan
        phi_raw[g] = rdg / denom if ok and np.isfinite(denom) and denom > 0 else np.nan
    finite = [v for v in phi_raw.values() if np.isfinite(v)]
    panel_median = float(np.median(finite)) if finite else 0.0
    phi = {g: (v if np.isfinite(v) else panel_median) for g, v in phi_raw.items()}
    phi_vals = np.array(list(phi.values()))
    var_phi = phi_vals.var(ddof=1) if len(phi_vals) > 1 else np.nan
    noise_var = bootstrap_phi_noise_floor(t1_raw, t2_raw, genes, thr1, thr2)
    floor = float(np.median(list(noise_var.values()))) if noise_var else 0.0
    w = max(0.0, 1.0 - floor / var_phi) if (np.isfinite(var_phi) and var_phi > 0) else 0.0
    return phi, w


# =================================================================================================
# D: real PIDC (Chan, Stumpf & Babtie 2017) -- adapted verbatim from
# preprocessing/run_competitor_methods.py's inline implementation (copied, not imported, because
# that script executes top-level analysis code on import).
# =================================================================================================
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
            a, loc, sc = __import__("scipy.stats", fromlist=["gamma"]).gamma.fit(v[v > 0], floc=0)
            F = lambda u, a=a, sc=sc: gamma_dist.cdf(u, a, loc=0, scale=sc)
        except Exception:
            mu, sd = float(np.mean(v)), float(np.std(v) or 1.0)
            F = lambda u, mu=mu, sd=sd: norm.cdf(u, mu, sd)
        for j in range(g):
            if i != j:
                D[i, j] += F(PUC[i, j])
    return pd.DataFrame(D, index=genes, columns=genes)


def compute_D(t2_raw, genes):
    """D: real PIDC on the larger sample (day4), all panel cells (not restricted to twin cells --
    D is a snapshot method, not a twin quantity)."""
    X = t2_raw[[f"{g}_mRNA" for g in genes]].to_numpy(dtype=float)
    return pidc_scores(X, genes)


# =================================================================================================
# Wz: optional twin-layer partial correlation (section 3.2)
#
# DISCLOSED interpretation: T's off-diagonal is not twin-restricted (any same-cell pair; the PDF
# only says S -- not T -- is "over twin cells", and Wz needs "thousands" of observations, which
# argues for using every day-4 cell, not only twin-eligible ones). B is the sister-sister matrix
# (same construction as C above) computed on one binomial half. The Schafer-Strimmer shrinkage
# intensity a* is computed with the standard ratio-of-variance-to-signal formula, using an
# ANALYTIC (delta-method) proxy for each entry's estimation variance -- 1/(meff_T-1) for T's
# contribution, 1/(meff_twin-1) for B's -- in place of the usual bootstrap/jackknife variance,
# since W = T - B mixes two differently-sampled statistics that don't share one common set of n
# aligned per-sample rows a bootstrap could resample jointly.
# =================================================================================================
def clone_weights_from_ids(clone_ids):
    counts = pd.Series(clone_ids).value_counts()
    return (1.0 / pd.Series(clone_ids).map(counts)).to_numpy()


def pooled_matrix(x_a, x_b, y_a, y_b, w):
    """Weighted correlation of x against y, pooling (x_a,y_b) with (x_b,y_a) at half weight each --
    used both for T (a/b = binomial halves) and, via sister_matrix_from_frame, for C (a/b=twins)."""
    w2 = np.concatenate([w, w]) / 2.0
    px = np.concatenate([x_a, x_b])
    py = np.concatenate([y_b, y_a])
    return weighted_spearman(px, py, w2)


def _one_split_half_W(X4, clone_ids, genes, genes_full, seed):
    """One binomial-thinning draw -> W = T - B for that draw. Factored out of compute_Wz so it can
    be repeated as a genuine bootstrap (see compute_Wz's 2026-09-18 fix note)."""
    rng = np.random.default_rng(seed)
    X4c = X4.tocoo()
    half1_data = rng.binomial(X4c.data.astype(np.int64), 0.5).astype(float)
    half2_data = X4c.data - half1_data
    from scipy.sparse import coo_matrix
    H1 = coo_matrix((half1_data, (X4c.row, X4c.col)), shape=X4c.shape).tocsr()
    H2 = coo_matrix((half2_data, (X4c.row, X4c.col)), shape=X4c.shape).tocsr()
    tot1 = np.asarray(H1.sum(axis=1)).ravel().astype(float)
    tot2 = np.asarray(H2.sum(axis=1)).ravel().astype(float)
    tot1[tot1 == 0], tot2[tot2 == 0] = 1.0, 1.0

    gidx = {g: i for i, g in enumerate(genes_full)}
    col = [gidx[g] for g in genes]
    p1 = np.log1p(H1[:, col].toarray() / tot1[:, None] * 1e4)
    p2 = np.log1p(H2[:, col].toarray() / tot2[:, None] * 1e4)

    w_cell = clone_weights_from_ids(clone_ids)
    T = np.full((len(genes), len(genes)), np.nan)
    for i, gx in enumerate(genes):
        for j in range(i, len(genes)):
            r = pooled_matrix(p1[:, i], p2[:, i], p1[:, j], p2[:, j], w_cell)
            T[i, j] = T[j, i] = r

    half1_df = pd.DataFrame(p1, columns=[f"{g}_mRNA" for g in genes])
    half1_df["clone_id"] = clone_ids
    B_df, meff_twin_ = sister_matrix_from_frame(half1_df, genes)
    return T - B_df.to_numpy(), meff_twin_


def compute_Wz(t2_raw, genes, gs, seed=SEED, n_boot=20):
    """Wz via a genuine bootstrap over the binomial split, not an analytic variance proxy.

    2026-09-18 fix: the original version used se = 1/sqrt(meff-1) (later also tried
    1/sqrt(meff-k-3)) as a single analytic stand-in for the partial correlation's true sampling
    noise. Checked against yscher's own real run on this exact panel (twinscore_mega_A.py's cached
    twin_layers/real/{run}.pkl output): her v (Wz's signal-share gate) is 0.00 for LARRY -- Wz
    carries no real signal there -- while the analytic-proxy version gave v=0.88-0.88, i.e. it was
    badly overconfident. Neither analytic tweak moved it, because the true source of the gap is
    that a SINGLE binomial split-half draw's noise isn't a good stand-in for the estimator's real
    sampling variance -- so this version draws n_boot=20 independent binomial splits, and gets both
    the shrinkage intensity and Wz's own standard error from their empirical spread across draws
    (a real bootstrap, matching what Schafer-Strimmer shrinkage is normally built from), rather
    than any single closed-form formula.
    """
    import scipy.io as sio
    t0 = time.time()
    X_full = sio.mmread(f"{SOURCE}/larry_qc_counts.mtx").tocsr()
    genes_full = np.array(open(f"{SOURCE}/genes.txt").read().split())
    obs = pd.read_csv(f"{SOURCE}/obs_metadata.csv", index_col=0)
    day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False).astype(int)
    mask = (day == T2).to_numpy()
    X4 = X_full[mask]
    obs4 = obs.loc[mask].reset_index()  # 'index' col = cell_id
    clone_ids = obs4["larry_clone_singletcode"].to_numpy()

    n = len(genes)
    W_draws, meff_twin_list = [], []
    for b in range(n_boot):
        Wb, meff_twin_b = _one_split_half_W(X4, clone_ids, genes, genes_full, seed=seed + b)
        W_draws.append(Wb)
        meff_twin_list.append(meff_twin_b)
    W_draws = np.stack(W_draws, axis=0)  # (n_boot, n, n)
    meff_twin_ = float(np.mean(meff_twin_list))
    meff_T = m_eff_step1(pd.Series(clone_ids).value_counts().to_numpy())
    log(f"    Wz: {n_boot} bootstrap split-half draws done ({time.time()-t0:.0f}s), "
        f"{X4.shape[0]:,} day4 cells")

    W_mean = W_draws.mean(axis=0)
    W_var_boot = W_draws.var(axis=0, ddof=1)  # real empirical sampling variance per entry
    iu = np.triu_indices(n, k=1)
    a_star = float(np.clip(W_var_boot[iu].sum() / np.sum(W_mean[iu] ** 2), 0.0, 1.0))

    def shrink_invert(Wm):
        Wshrunk = (1 - a_star) * Wm + a_star * np.eye(n)
        P = -np.linalg.pinv(Wshrunk + 1e-8 * np.eye(n))
        d = np.sqrt(np.clip(np.outer(np.diag(P), np.diag(P)), 1e-12, None))
        PC = P / d
        np.fill_diagonal(PC, 0.0)
        return PC

    PC_mean = shrink_invert(W_mean)
    PC_draws = np.stack([shrink_invert(W_draws[b]) for b in range(n_boot)], axis=0)
    se = PC_draws.std(axis=0, ddof=1)  # real bootstrap SE of the shrunk partial correlation
    Wz = pd.DataFrame(np.where(se > 1e-9, PC_mean / np.where(se > 1e-9, se, 1.0), 0.0),
                       index=genes, columns=genes)
    v = signal_share(Wz.to_numpy()[~np.eye(n, dtype=bool)])
    log(f"    Wz: a*={a_star:.3f}  v={v:.3f}  (meff_T={meff_T:.0f}, meff_twin={meff_twin_:.0f})")
    return Wz, v, a_star


# =================================================================================================
# main per-gene-set pipeline
# =================================================================================================
def run_gene_set(gs, ce_edges, compute_wz=True):
    log(f"[{gs}] loading raw twin-pair data")
    t1_raw, t2_raw, genes = load_raw(gs)
    n_genes = len(genes)

    log(f"[{gs}] real-permutation S, z_het, direction terms (yscher's own run, real draws)")
    real = yscher_real_permutation_terms(gs, genes)
    S_t1, S_t2 = real["S_t1"], real["S"]
    zhet_matrix = real["zhet"]

    log(f"[{gs}] C, heritability at t1/t2 (fresh, real weighted-Spearman, no permutation needed)")
    C_t1, meff_twin_t1 = sister_matrix_from_frame(t1_raw, genes)
    C_t2, meff_twin_t2 = sister_matrix_from_frame(t2_raw, genes)
    h_t1 = {g: C_t1.loc[g, g] for g in genes}
    h_t2 = {g: C_t2.loc[g, g] for g in genes}

    log(f"[{gs}] cross-sample twin correlation (persistence numerator; direction reused from yscher)")
    rho_dagger, meff_cross_ = cross_matrix(t1_raw, t2_raw, genes)
    rho_dagger_diag = {g: rho_dagger.loc[g, g] for g in genes}

    log(f"[{gs}] persistence phi_g, reliability w")
    phi, w_rel = persistence(h_t1, h_t2, rho_dagger_diag, genes, meff_twin_t1, meff_twin_t2, meff_cross_,
                              t1_raw, t2_raw)

    log(f"[{gs}] zreg, R, twin-signal gate g")
    # sd_reg: zreg = S - lambda*C is a NEW supplement-only combination (S minus the sister matrix,
    # not yscher's own signed-flux z_reg from the manuscript pipeline) -- she never computes this
    # exact quantity, so there is no real-permutation draw to substitute here. Kept as the repo's
    # existing disclosed analytic proxy (analytic_zscores.py: "z_reg_gated's null has Y-sibling
    # -block structure ... SD_HET_T1 is a proxy").
    sd_reg = DEFAULT_SD["het_t1"]
    sd_twin_t2 = null_sd(meff_twin_t2)
    lam = np.minimum(1.0, zhet_matrix.abs() / 2.33)
    zreg = (S_t2 - lam * C_t2) / sd_reg
    R_fn = clr_calibrate(zreg, genes)
    zC = C_t2 / sd_twin_t2
    gate_g = signal_share(zC.to_numpy()[~np.eye(n_genes, dtype=bool)])

    if compute_wz:
        log(f"[{gs}] D (real PIDC) and Wz (split-half partial correlation)")
    else:
        log(f"[{gs}] D (real PIDC)")
    D_mat = compute_D(t2_raw, genes)
    if compute_wz:
        Wz_mat, v_gate, a_star = compute_Wz(t2_raw, genes, gs)
    else:
        Wz_mat = pd.DataFrame(0.0, index=genes, columns=genes)
        v_gate, a_star = 0.0, np.nan

    log(f"[{gs}] direction: z^dagger (yscher's real z_fwd/z_rev), gamma, kappa_gamma, q")
    U = [(a, b) for a in genes for b in genes if a != b]
    z_dag_xy = np.array([real["z_dagger"].loc[a, b] for a, b in U])
    z_dag_yx = np.array([real["z_dagger"].loc[b, a] for a, b in U])
    gamma_raw = (np.abs(z_dag_xy) - np.abs(z_dag_yx)) / np.sqrt(2 * (1 - 2 / np.pi))
    kappa_gamma = signal_share(gamma_raw)
    q = 0.5 + (R0 - 0.5) * (2 * norm.cdf(np.abs(gamma_raw)) - 1) * np.sign(gamma_raw)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    gate_dir = (np.maximum(np.abs(z_dag_xy), np.abs(z_dag_yx)) > Z_DIRECTION_GATE).astype(float)
    direction_term = kappa_gamma * gate_dir * np.log(q)

    log(f"[{gs}] assembling PAIR and TwinScore")
    D_vec = np.array([D_mat.loc[a, b] for a, b in U])
    R_vec = np.array([R_fn(a, b) for a, b in U])
    Wz_vec = np.array([Wz_mat.loc[a, b] for a, b in U])
    PAIR = s(D_vec) + gate_g * s(R_vec) + gate_g * v_gate * s(Wz_vec)
    phi_x = np.array([phi[a] for a, b in U])
    score = w_rel * s(phi_x) + s(PAIR) + direction_term

    # -------------------------------------------------------------- scenario labels (section 5)
    # DISCLOSED cutoffs (labeling convenience only, does not affect the score): stage I uses
    # yscher's own real z_rho_t1/z_rho_t2 against her own existence-filter threshold (2.576, per
    # her twinscore_table.py docstring: "existence |z_rho(t1)|>2.576 OR |z_rho(t2)|>2.576"). Stage
    # II uses the real z_het above. "phi approx 1" vs "phi<1" uses phi>=0.5; z_stable (no real
    # -permutation equivalent in her outputs) keeps the analytic meff-based null sd.
    meff_step1_t1 = m_eff_step1(t1_raw.groupby("clone_id").size().to_numpy())
    meff_step1_t2 = m_eff_step1(t2_raw.groupby("clone_id").size().to_numpy())
    sd_change = np.sqrt(null_sd(meff_step1_t1) ** 2 + null_sd(meff_step1_t2) ** 2)
    stage1 = np.array([max(abs(real["z_rho_t1"].loc[a, b]), abs(real["z_rho_t2"].loc[a, b])) > Z_DIRECTION_GATE
                        for a, b in U])
    stage2 = np.array([abs(zhet_matrix.loc[a, b]) > Z_ONE_SIDED for a, b in U])
    r_present = np.abs(R_vec) > Z_ONE_SIDED
    z_stable = np.array([(S_t2.loc[a, b] - S_t1.loc[a, b]) / sd_change for a, b in U])
    phi_pair = np.array([min(phi[a], phi[b]) for a, b in U])
    inherited = (phi_pair >= 0.5) & (np.abs(z_stable) <= Z_ONE_SIDED)

    def scenario(i):
        if not stage1[i]:
            return "1_no_regulation_single_state"
        if not stage2[i]:
            return "2_regulation_single_state"
        if inherited[i]:
            return "3_regulation_on_inherited_state" if r_present[i] else "4_no_regulation_inherited_state_only"
        return "6_regulation_during_differentiation" if r_present[i] else "5_no_regulation_differentiation_only"
    scen = [scenario(i) for i in range(len(U))]

    pair_df = pd.DataFrame({
        "gene_1": [a for a, b in U], "gene_2": [b for a, b in U],
        "S": [S_t2.loc[a, b] for a, b in U], "C": [C_t2.loc[a, b] for a, b in U],
        "zhet": [zhet_matrix.loc[a, b] for a, b in U], "lambda": [lam.loc[a, b] for a, b in U],
        "zreg": [zreg.loc[a, b] for a, b in U], "R": R_vec, "gate_g": gate_g,
        "D": D_vec, "Wz": Wz_vec, "gate_v": v_gate,
        "rho_dagger_xy": [real["rho_dagger"].loc[a, b] for a, b in U],
        "rho_dagger_yx": [real["rho_dagger"].loc[b, a] for a, b in U],
        "z_dagger_xy": z_dag_xy, "z_dagger_yx": z_dag_yx, "gamma": gamma_raw, "kappa_gamma": kappa_gamma,
        "q": q, "direction_term": direction_term, "PAIR": PAIR, "TwinScore": score,
        "scenario": scen, "collectri_edge": [1 if p in ce_edges else 0 for p in U],
    })
    gene_df = pd.DataFrame({
        "gene": genes, "h_t1": [h_t1[g] for g in genes], "h_t2": [h_t2[g] for g in genes],
        "rho_dagger_gg": [rho_dagger_diag[g] for g in genes], "phi": [phi[g] for g in genes],
    })
    diagnostics = dict(gs=gs, n_genes=n_genes, n_pairs=len(U), meff_twin_t1=meff_twin_t1,
                        meff_twin_t2=meff_twin_t2, meff_cross=meff_cross_, meff_step1_t2=meff_step1_t2,
                        w=w_rel, gate_g=gate_g, gate_v=v_gate, kappa_gamma=kappa_gamma,
                        wz_shrinkage_a=a_star, median_h=float(np.median(list(h_t2.values()))))
    return pair_df, gene_df, diagnostics


# =================================================================================================
# evaluation (section 6 style)
# =================================================================================================
def full_report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    k = int(y.sum())
    order = np.argsort(-sc, kind="stable")
    top_k = order[:k]
    tp = int((y[top_k] == 1).sum())
    try:
        auroc = roc_auc_score(y, sc)
    except ValueError:
        auroc = float("nan")
    return dict(auprc=auprc, auprc_random=rand, auprc_x=auprc / rand if rand > 0 else float("nan"),
                auroc=auroc, prec=tp / k if k else float("nan"), rec=tp / k if k else float("nan"), tp=tp, k=k)


YSCHER_NETWORKS = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/networks"


def load_competitor_allgenes(method, gs, univ, genes):
    """Best-of run for one competitor method, read from yscher's own exports/networks/*.csv --
    mirrors her helpers/twinscore_table.py exactly: only a file that scored the EXACT full
    ordered universe of this panel (every gene x every other gene) is admitted."""
    stem = {"rho": "rho", "ppcor": "ppcor", "pidc": "pidc", "genie3": "genie3", "grnboost2": "grnboost"}[method]
    full = set(univ)
    best = None
    for f in sorted(glob.glob(f"{YSCHER_NETWORKS}/*.csv")):
        if stem not in os.path.basename(f).lower():
            continue
        try:
            d0 = pd.read_csv(f)
        except Exception:
            continue
        c = {x.lower(): x for x in d0.columns}
        cs = c.get("tf") or c.get("source") or d0.columns[0]
        ct = c.get("target") or d0.columns[1]
        cw = c.get("importance") or c.get("weight") or c.get("score") or d0.columns[2]
        d0 = d0[[cs, ct, cw]].dropna()
        d0.columns = ["TF", "target", "w"]
        if set(zip(d0.TF, d0.target)) != full:
            continue
        m = {(a, b): float(v) for a, b, v in d0.itertuples(index=False)}
        best = np.array([m[p] for p in univ], float)
        break
    return best


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    summary_rows = []
    for gs in GENE_SETS:
        t0 = time.time()
        pair_df, gene_df, diag = run_gene_set(gs, CE, compute_wz=True)
        pair_df.to_csv(f"{HERE}/{gs}_pair_terms.csv", index=False)
        gene_df.to_csv(f"{HERE}/{gs}_gene_terms.csv", index=False)

        U = list(zip(pair_df.gene_1, pair_df.gene_2))
        y = pair_df.collectri_edge.to_numpy()
        rep = full_report(pair_df.TwinScore.to_numpy(), y)
        pair_rep = full_report(pair_df.PAIR.to_numpy(), y)

        genes = sorted(set(a for a, b in U) | set(b for a, b in U))
        methods = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
        best_name, best_m = None, None
        for m in methods:
            sc = load_competitor_allgenes(m, gs, U, genes)
            if sc is None or not np.isfinite(sc).any():
                continue
            cm = full_report(sc, y)
            if best_m is None or cm["auprc_x"] > best_m["auprc_x"]:
                best_name, best_m = m, cm

        log(f"[{gs}] n_pairs={len(U)} n_true={int(y.sum())} "
            f"TwinScore={rep['auprc_x']:.3f}x PAIR-alone={pair_rep['auprc_x']:.3f}x "
            f"best_comp={best_name}({best_m['auprc_x']:.3f}x)  w={diag['w']:.3f} g={diag['gate_g']:.3f} "
            f"v={diag['gate_v']:.3f} kappa_gamma={diag['kappa_gamma']:.3f} "
            f"({time.time()-t0:.0f}s)")

        summary_rows.append(dict(
            gene_set=gs, n_pairs=len(U), n_true=int(y.sum()),
            calls=best_m["k"], hits=rep["tp"], prec_pct=100 * rep["prec"], rec_pct=100 * rep["rec"],
            auroc=rep["auroc"], auprc=rep["auprc"], auprc_x=rep["auprc_x"],
            pair_alone_auprc_x=pair_rep["auprc_x"],
            best_competitor=best_name, comp_auprc_x=best_m["auprc_x"] if best_m else np.nan,
            w=diag["w"], gate_g=diag["gate_g"], gate_v=diag["gate_v"], kappa_gamma=diag["kappa_gamma"],
            wz_shrinkage_a=diag["wz_shrinkage_a"], median_h=diag["median_h"],
        ))

    summary = pd.DataFrame(summary_rows)
    # SUMMARY_SUFFIX: lets parallel jobs each write their own file (e.g. run_id) instead of
    # racing on the shared summary CSV via this read-modify-write (no locking); merge afterward.
    suffix = os.environ.get("SUMMARY_SUFFIX", "")
    out_path = f"{HERE}/twinscore_supplement_larry_summary{('_' + suffix) if suffix else ''}.csv"
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        prior = prior[~prior.gene_set.isin(summary.gene_set)]
        summary = pd.concat([prior, summary], ignore_index=True)
    summary.to_csv(out_path, index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
