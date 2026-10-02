# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Full TwinScore-supplement formula, DIRECTED, for every ordered gene pair, on BoolODE-simulated
HSC / mCAD twin data (twin_final_states.csv). Adapted from two reference implementations:

  - apply_twinscore_supplement_larry.py (real LARRY data): the full-fidelity Wz bootstrap and PIDC
    D implementation are adapted near-verbatim from here (binomial split-half bootstrap for Wz,
    Bayesian-blocks-discretised PIDC for D), and the phi/w persistence-reliability bootstrap is
    reused essentially verbatim (bootstrap_phi_noise_floor).
  - apply_twinscore_supplement.py (Gillespie e13_pos100 benchmark): the "one twin pair per clone,
    every clone weight uniform" data shape (clone_id=pair_id, exactly 2 cells/clone at every
    timepoint) is identical to this BoolODE twin data, so persistence's raw-CSV-driven
    implementation is the closest structural precedent.

Data: simulation_data/boolode_sims_replicates/{HSC,mCAD}/replicate_{0,1,2}/twin_final_states.csv.
Columns: branch_tp, <genes...>, pair_id, source, step, t. source in {trunk, twin_A, twin_B}; trunk
rows (a full 0->final reference lineage) are dropped. pair_id is the twin clone id. t1 = first
post-branch saved step (500 for HSC / 300 for mCAD), t2 = final saved step (799 / 499).

Values are CONTINUOUS non-negative BoolODE floats, not real molecule counts.

  - All CORRELATION-based terms (S, C/heritability, phi, w, z_het, R/z_reg_gated, rho_cross/
    z_dagger/gamma/kappa_gamma/q) are computed on the RAW CONTINUOUS values. Spearman correlation
    is rank-based and does not need integer counts; rounding these terms would only destroy
    information (many close-but-distinct floats would become tied ranks) for no benefit.
  - D (PIDC) is still computed on values ROUNDED TO THE NEAREST NON-NEGATIVE INTEGER (per the
    user's original explicit instruction; PIDC's discretisation mechanic treats each cell's value
    as a count of discrete outcomes). This is flagged clearly as a real methodological stretch --
    BoolODE values are O(0.001)-O(3.5), so a large fraction round to 0 (measured and reported
    below).
  - Wz (shrinkage partial correlation) was CHANGED on 2026-09-21 per explicit user instruction to
    use the REAL CONTINUOUS (unrounded) t2 values, via a clone-level with-replacement resampling
    bootstrap (not binomial thinning, which is a count-only operation and was catastrophic for HSC
    -- see compute_Wz's docstring for the full rationale and handoff/
    HANDOFF_2026-09-20_twinscore_supplement.md, Bug #1, for the precedent this mirrors). D remains
    on the rounded-integer path unchanged -- the user asked only about Wz.

Ground truth: simulation_data/twinfer_format/{net}/interaction_matrix.csv (BoolODE's own network,
no header row -- gene order given by the sibling gene_order.txt file), DIRECTED (rows=source,
cols=target, nonzero=edge).

Formula (handoff/HANDOFF_2026-09-20_twinscore_supplement.md):
  TwinScore(x->y) = w*s(phi_x) + s(PAIR) + kappa_gamma * 1[gate] * log(q(x->y))
  PAIR = s(D) + g*s(R) + g*v*s(Wz)
  phi_x = rho_dagger(x,x) / sqrt(h_x(t1)*h_x(t2))

Directionality: PAIR (D, R, Wz) is symmetric in (x,y) by construction (CLR-calibrated symmetric
inputs). All direction comes from phi_x (source-gene-only) and the gamma/kappa_gamma/q term (built
from z_dagger(x->y) vs z_dagger(y->x), genuinely asymmetric).

ALL ordered pairs are scored (not a pre-gated subset): the gate that infer_with_twinfer's own
stage3/gated_regulation pipeline applies (drops Gata2-type zero-variance genes, ~45/55 unordered
pairs surviving for HSC) is bypassed here by calling the lower-level per-pair correlation
primitives directly on every gene pair. A handful of pairs (those touching a literally
zero-variance gene at some sample) will have undefined (NaN) Spearman correlations; these are
handled by the same NaN-fallback conventions the LARRY script uses (phi -> panel median; s(.) sends
NaN entries to the bottom of the ranked scale) and are flagged in the summary report.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma as gamma_dist
from sklearn.metrics import average_precision_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, split_twins, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
    get_unit_weights, weighted_spearman,
)

SIM_ROOT = f'{TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims_replicates'
GT_ROOT = f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/clean_code/benchmarks/boolode/gillespie_comparison'

NETS = ["HSC", "mCAD"]
REPS = [0, 1, 2]
SEED = 0
N_RAND_HET = 40       # fresh random-pair reference draws for z_het centre (matches LARRY script)
N_BOOT_PHI = 20        # clone-level cluster bootstrap draws for phi's noise floor
N_BOOT_WZ = 20          # binomial split-half bootstrap draws for Wz
R0 = 0.69
Z_DIRECTION_GATE = 2.576


def log(msg):
    print(msg, flush=True)


# =================================================================================================
# small shared helpers (verbatim from apply_twinscore_supplement_larry.py)
# =================================================================================================
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


def m_eff_step1(clone_sizes):
    n = np.asarray(clone_sizes, float)
    return float(len(n) ** 2 / (1.0 / n).sum())


def m_eff_twin(clone_sizes):
    n = np.asarray(clone_sizes, float)
    n = n[n >= 2]
    c = n * (n - 1) / 2.0
    return float(len(n) ** 2 / (1.0 / c).sum())


def m_eff_cross(n1, n2):
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    return float(len(n1) ** 2 / (1.0 / (n1 * n2)).sum())


# =================================================================================================
# data loading
# =================================================================================================
def load_twin_data(net, rep):
    path = f"{SIM_ROOT}/{net}/replicate_{rep}/twin_final_states.csv"
    df = pd.read_csv(path)
    genes = [c for c in df.columns if c not in ("branch_tp", "pair_id", "source", "step", "t")]
    df = df[df.source.isin(["twin_A", "twin_B"])].reset_index(drop=True)
    branch_tp = int(df.branch_tp.iloc[0])
    steps = sorted(df.step.unique())
    # Drop the trivial branch-point step itself: twin_A/twin_B are identical there by
    # construction (pearson=1.0 exactly, since they haven't diverged yet), which would trivially
    # inflate heritability/phi if used as t1. t1 = first POST-branch saved step, t2 = final step.
    post_branch_steps = [t for t in steps if t > branch_tp]
    t1, t2 = post_branch_steps[0], post_branch_steps[-1]
    return df, genes, int(t1), int(t2)


def canonical_frame(df, genes, t):
    """One row per (pair_id, twin_source) cell at step t, canonical clone_id/cell_id/gene_mRNA
    table the package's correlation_functions.py primitives expect."""
    d = df[df.step == t].reset_index(drop=True).copy()
    out = pd.DataFrame({
        "cell_id": d.pair_id.astype(str) + "_" + d.source,
        "clone_id": d.pair_id,
    })
    for g in genes:
        out[f"{g}_mRNA"] = d[g].to_numpy(dtype=float)
    return out


def load_ground_truth(net):
    genes = [l.strip() for l in open(f"{GT_ROOT}/{net}/gene_order.txt") if l.strip()]
    M = pd.read_csv(f"{GT_ROOT}/{net}/interaction_matrix.csv", header=None).to_numpy(dtype=float)
    assert M.shape == (len(genes), len(genes)), (M.shape, len(genes))
    true_edges = set()
    for i, a in enumerate(genes):
        for j, b in enumerate(genes):
            if i != j and M[i, j] != 0:
                true_edges.add((a, b))   # row=source a, col=target b : a -> b
    return genes, true_edges


# =================================================================================================
# S(x,y), C(x,y)/heritability (sister matrix), rho_dagger (cross-time twin matrix)
# =================================================================================================
def sister_matrix_from_frame(raw, genes):
    """C(x,y): sister-sister cross-gene correlation (twin_A vs twin_B), both orderings pooled.
    Diagonal C(g,g) is heritability h_g at this sample. Verbatim logic from the LARRY script,
    generic over any clone_id/twin structure (here: exactly one twin per clone)."""
    tw = assign_twin_id(raw)
    twin_0, twin_1 = split_twins(tw)
    w = get_unit_weights(twin_0, unit="clone")
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


def cross_matrix(t1_raw, t2_raw, genes):
    """rho^dagger(x,y): x in the t1 twin against y in its t2 clone-mate, directed.
    Diagonal = rho^dagger_gg, the persistence numerator."""
    at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
    ordered = [(a, b) for a in genes for b in genes if a != b]
    xc = get_cross_correlations(at1, at2, gene_pairs=ordered, unit="clone")
    both = set(t1_raw.clone_id) & set(t2_raw.clone_id)
    n1 = t1_raw[t1_raw.clone_id.isin(both)].groupby("clone_id").size()
    n2 = t2_raw[t2_raw.clone_id.isin(both)].groupby("clone_id").size()
    both = sorted(both)
    meff = m_eff_cross(n1.loc[both].to_numpy(), n2.loc[both].to_numpy())
    return xc, meff


def het_stats(raw, genes, n_rand=N_RAND_HET, seed=SEED):
    """rho_Delta(x,y) observed and its random-pairing reference centre (mean of n_rand fresh
    draws), on the SAME sample. z_het is assembled from these plus a data-appropriate (m_eff-
    based) null SD downstream, not a LARRY-calibrated constant."""
    tw = assign_twin_id(raw)
    rho_delta_obs, _ = calculate_twin_random_correlations(raw, tw, genes, random_state=seed, unit="clone")
    draws = [calculate_twin_random_correlations(raw, tw, genes, random_state=seed + 100 + i, unit="clone")[1].to_numpy()
             for i in range(n_rand)]
    rho_delta_random = pd.DataFrame(np.mean(draws, axis=0), index=genes, columns=genes)
    return rho_delta_obs, rho_delta_random


# =================================================================================================
# persistence phi_g and its reliability w -- verbatim bootstrap logic from the LARRY script
# =================================================================================================
def bootstrap_phi_noise_floor(t1_raw, t2_raw, genes, thr1, thr2, n_boot=N_BOOT_PHI, seed=SEED):
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


def persistence(h1, h2, rho_dagger, genes, meff_twin_t1, meff_twin_t2, t1_raw, t2_raw):
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
    return phi, w, phi_raw


# =================================================================================================
# D: real PIDC, adapted verbatim from the LARRY script, run on ROUNDED-to-integer t2 values
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


def compute_D(t2_int, genes):
    X = t2_int[genes].to_numpy(dtype=float)
    return pidc_scores(X, genes)


# =================================================================================================
# Wz: twin-layer shrinkage partial correlation via a CLONE-LEVEL RESAMPLING bootstrap on the REAL
# CONTINUOUS (unrounded) t2 BoolODE values. Changed 2026-09-21 per explicit user instruction: the
# original implementation (still used for D/PIDC) rounded values to integer pseudo-counts and
# thinned them binomially (Binomial(round(value), 0.5)) to create synthetic split-half technical
# replicates -- a mechanic that is fundamentally count-only and was catastrophic for HSC (values are
# mostly O(0.001)-O(1), so 44-100% of per-gene values rounded to exactly 0, crushing the bootstrap
# to a*=1.0 / v=0.00 for every HSC replicate). Binomial thinning cannot be applied to continuous
# values at all, so per the user's instructions (mirroring how bootstrap_phi_noise_floor already
# resamples CLONES, not counts, for phi's reliability bootstrap, and how Bug #1 in
# handoff/HANDOFF_2026-09-20_twinscore_supplement.md fixed Wz's SE for the LARRY script by moving
# from an analytic formula to a genuine bootstrap) each of the N_BOOT_WZ draws now:
#   1. resamples pair_id/clone_id WITH REPLACEMENT (clone-level nonparametric bootstrap, exactly the
#      same resampling family as bootstrap_phi_noise_floor -- draws are over cells/clones, not over
#      counts), keeping each resampled clone's twin_A/twin_B pair intact and re-indexing clone_id so
#      repeated draws of the same original clone don't collide;
#   2. recomputes T (the total pairwise gene-gene correlation across all resampled cells) via the
#      SAME primitive used for S (calculate_pairwise_gene_gene_correlation_matrix, clone-weighted
#      Spearman) and B (the sister/twin correlation on the resampled data) via the SAME primitive
#      used for C (sister_matrix_from_frame), on the raw continuous values -- no rounding, no
#      thinning, no log1p needed since Spearman is rank-based and scale-invariant;
#   3. W_draw = T - B, exactly the same raw-correlation-matrix definition as before.
# The Schafer-Strimmer shrinkage step downstream (shrinkage intensity a* and SE from the empirical
# variance of the W_draws, shrink-toward-identity, pseudo-inverse, partial-correlation z-score) is
# UNCHANGED -- that formula never cared whether its inputs were counts or continuous, only the
# bootstrap MECHANISM that generates each draw's noisy realization of W has changed.
# =================================================================================================
def _one_bootstrap_W_continuous(t2_raw, genes, seed):
    rng = np.random.default_rng(seed)
    clone_idx = t2_raw.groupby("clone_id").indices
    universe = sorted(clone_idx.keys())
    drawn = rng.choice(universe, size=len(universe), replace=True)
    idx_parts, cid_parts = [], []
    for i, c in enumerate(drawn):
        r = clone_idx[c]
        idx_parts.append(r)
        cid_parts.append(np.full(len(r), i))
    boot = t2_raw.iloc[np.concatenate(idx_parts)].reset_index(drop=True).copy()
    boot["clone_id"] = np.concatenate(cid_parts)
    # cell_id must be unique per row for calculate_pairwise_gene_gene_correlation_matrix's
    # duplicate-cell_id guard; the original cell_id strings collide whenever a clone is drawn more
    # than once, so replace with a fresh per-row index (cell_id's identity is not otherwise used).
    boot["cell_id"] = np.arange(len(boot)).astype(str)

    T = calculate_pairwise_gene_gene_correlation_matrix(boot, genes, use_clone=True)
    B, _ = sister_matrix_from_frame(boot, genes)
    return (T - B).to_numpy()


def compute_Wz(t2_raw, genes, seed=SEED, n_boot=N_BOOT_WZ):
    t0 = time.time()
    n = len(genes)
    W_draws = [_one_bootstrap_W_continuous(t2_raw, genes, seed=seed + b) for b in range(n_boot)]
    W_draws = np.stack(W_draws, axis=0)

    W_mean = W_draws.mean(axis=0)
    W_var_boot = W_draws.var(axis=0, ddof=1)
    iu = np.triu_indices(n, k=1)
    denom = np.sum(W_mean[iu] ** 2)
    a_star = float(np.clip(W_var_boot[iu].sum() / denom, 0.0, 1.0)) if denom > 0 else 1.0

    def shrink_invert(Wm):
        Wm = np.nan_to_num(Wm, nan=0.0)
        Wshrunk = (1 - a_star) * Wm + a_star * np.eye(n)
        P = -np.linalg.pinv(Wshrunk + 1e-8 * np.eye(n))
        d = np.sqrt(np.clip(np.outer(np.diag(P), np.diag(P)), 1e-12, None))
        PC = P / d
        np.fill_diagonal(PC, 0.0)
        return PC

    PC_mean = shrink_invert(W_mean)
    PC_draws = np.stack([shrink_invert(W_draws[b]) for b in range(n_boot)], axis=0)
    se = PC_draws.std(axis=0, ddof=1)
    Wz = pd.DataFrame(np.where(se > 1e-9, PC_mean / np.where(se > 1e-9, se, 1.0), 0.0),
                       index=genes, columns=genes)
    v = signal_share(Wz.to_numpy()[~np.eye(n, dtype=bool)])
    log(f"    Wz: a*={a_star:.3f}  v={v:.3f}  ({time.time()-t0:.0f}s)")
    return Wz, v, a_star


# =================================================================================================
# main per-network/replicate pipeline
# =================================================================================================
def run_one(net, rep):
    log(f"[{net} rep{rep}] loading twin data")
    df, genes, t1, t2 = load_twin_data(net, rep)
    n_genes = len(genes)
    t1_raw = canonical_frame(df, genes, t1)
    t2_raw = canonical_frame(df, genes, t2)
    log(f"[{net} rep{rep}] t1={t1} t2={t2} n_genes={n_genes} n_clones={t1_raw.clone_id.nunique()}")

    # ---- rounding for the count-dependent D/Wz components ----------------------------------
    t2_int = t2_raw.copy()
    round_frac_zero = {}
    for g in genes:
        vals = t2_raw[f"{g}_mRNA"].to_numpy()
        rounded = np.rint(np.clip(vals, 0, None))
        t2_int[g] = rounded
        round_frac_zero[g] = float((rounded == 0).mean())
    log(f"[{net} rep{rep}] fraction of rounded t2 values == 0, per gene: "
        f"{ {g: round(v,3) for g,v in round_frac_zero.items()} }")

    # ---- S (gene-gene rho), C (sister/heritability), rho_dagger (cross-twin) ---------------
    log(f"[{net} rep{rep}] S, C/heritability, rho_dagger")
    S_t2 = calculate_pairwise_gene_gene_correlation_matrix(t2_raw, genes, use_clone=True)
    C_t1, meff_twin_t1 = sister_matrix_from_frame(t1_raw, genes)
    C_t2, meff_twin_t2 = sister_matrix_from_frame(t2_raw, genes)
    h_t1 = {g: C_t1.loc[g, g] for g in genes}
    h_t2 = {g: C_t2.loc[g, g] for g in genes}
    rho_dagger, meff_cross_ = cross_matrix(t1_raw, t2_raw, genes)
    rho_dagger_diag = {g: rho_dagger.loc[g, g] for g in genes}

    log(f"[{net} rep{rep}] persistence phi_g, reliability w")
    phi, w_rel, phi_raw = persistence(h_t1, h_t2, rho_dagger_diag, genes, meff_twin_t1, meff_twin_t2,
                                       t1_raw, t2_raw)
    log(f"    phi (raw, pre-median-fill): { {g: (round(v,3) if np.isfinite(v) else None) for g,v in phi_raw.items()} }")
    log(f"    w={w_rel:.3f}")

    # ---- z_het, lambda, R = CLR(zreg) --------------------------------------------------------
    log(f"[{net} rep{rep}] z_het, zreg, R")
    rho_delta_obs, rho_delta_random = het_stats(t2_raw, genes)
    clone_sizes_t2 = t2_raw.groupby("clone_id").size().to_numpy()
    sd_het = null_sd(m_eff_twin(clone_sizes_t2))
    zhet_matrix = (rho_delta_obs - rho_delta_random) / sd_het

    sd_reg = null_sd(meff_twin_t2)   # dataset-appropriate proxy for zreg's null (same role as
                                       # analytic_zscores.SD_HET_T1 in the reference scripts, but
                                       # derived from this dataset's own clone structure)
    lam = np.minimum(1.0, zhet_matrix.abs() / 2.33)
    zreg = (S_t2 - lam * C_t2) / sd_reg
    R_fn = clr_calibrate(zreg, genes)
    zC = C_t2 / null_sd(meff_twin_t2)
    gate_g = signal_share(zC.to_numpy()[~np.eye(n_genes, dtype=bool)])

    # ---- D (PIDC on rounded integer t2 values) -----------------------------------------------
    log(f"[{net} rep{rep}] D (PIDC, rounded-integer t2 values)")
    D_mat = compute_D(t2_int, genes)

    # ---- Wz (clone-level resampling bootstrap on REAL CONTINUOUS, unrounded t2 values) --------
    log(f"[{net} rep{rep}] Wz (clone-resampling bootstrap partial correlation, continuous t2 values)")
    Wz_mat, v_gate, a_star = compute_Wz(t2_raw, genes)

    # ---- direction: z_dagger(x->y) vs z_dagger(y->x), gamma, kappa_gamma, q ------------------
    log(f"[{net} rep{rep}] direction: z_dagger, gamma, kappa_gamma, q")
    U = [(a, b) for a in genes for b in genes if a != b]
    sd_cross = null_sd(meff_cross_)
    z_dag_xy = np.array([rho_dagger.loc[a, b] for a, b in U]) / sd_cross
    z_dag_yx = np.array([rho_dagger.loc[b, a] for a, b in U]) / sd_cross
    gamma_raw = (np.abs(z_dag_xy) - np.abs(z_dag_yx)) / np.sqrt(2 * (1 - 2 / np.pi))
    kappa_gamma = signal_share(gamma_raw)
    q = 0.5 + (R0 - 0.5) * (2 * norm.cdf(np.abs(gamma_raw)) - 1) * np.sign(gamma_raw)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    gate_dir = (np.maximum(np.abs(z_dag_xy), np.abs(z_dag_yx)) > Z_DIRECTION_GATE).astype(float)
    direction_term = kappa_gamma * gate_dir * np.log(q)

    # ---- assemble PAIR and TwinScore ----------------------------------------------------------
    log(f"[{net} rep{rep}] assembling PAIR and TwinScore")
    D_vec = np.array([D_mat.loc[a, b] for a, b in U])
    R_vec = np.array([R_fn(a, b) for a, b in U])
    Wz_vec = np.array([Wz_mat.loc[a, b] for a, b in U])
    PAIR = s(D_vec) + gate_g * s(R_vec) + gate_g * v_gate * s(Wz_vec)
    phi_x = np.array([phi[a] for a, b in U])
    score = w_rel * s(phi_x) + s(PAIR) + direction_term

    # ---- ground truth, evaluation ---------------------------------------------------------
    gt_genes, true_edges = load_ground_truth(net)
    assert set(gt_genes) == set(genes), (set(gt_genes) - set(genes), set(genes) - set(gt_genes))
    y = np.array([1 if p in true_edges else 0 for p in U])

    pair_df = pd.DataFrame({
        "source_gene": [a for a, b in U], "target_gene": [b for a, b in U],
        "phi_x": phi_x, "w": w_rel, "h1": [h_t1[a] for a, b in U], "h2": [h_t2[a] for a, b in U],
        "D": D_vec, "R": R_vec, "Wz": Wz_vec,
        "gamma": gamma_raw, "kappa_gamma": kappa_gamma, "q": q,
        "is_true_edge": y, "TwinScore": score,
    })
    out_path = f"{OUT_DIR}/twinscore_directed_{net}_rep{rep}.csv"
    pair_df.to_csv(out_path, index=False)
    log(f"[{net} rep{rep}] wrote {out_path}  ({len(pair_df)} directed pairs)")

    finite_score = np.nan_to_num(score, nan=np.nanmin(score) - 1.0 if np.isfinite(score).any() else 0.0)
    auprc = average_precision_score(y, finite_score)
    base_rate = float(y.mean())
    k = int(y.sum())
    order = np.argsort(-finite_score, kind="stable")
    top_k = order[:k]
    prec_at_k = float((y[top_k] == 1).sum()) / k if k else float("nan")

    return dict(net=net, rep=rep, n_genes=n_genes, n_pairs=len(U), n_true=k,
                auprc=auprc, base_rate=base_rate, auprc_x=auprc / base_rate if base_rate > 0 else np.nan,
                prec_at_k=prec_at_k, w=w_rel, gate_g=gate_g, gate_v=v_gate, kappa_gamma=kappa_gamma,
                wz_shrinkage_a=a_star,
                frac_rounded_zero_mean=float(np.mean(list(round_frac_zero.values()))),
                n_nan_phi_raw=int(sum(1 for v in phi_raw.values() if not np.isfinite(v))))


def main():
    rows = []
    for net in NETS:
        for rep in REPS:
            t0 = time.time()
            r = run_one(net, rep)
            r["runtime_sec"] = time.time() - t0
            rows.append(r)
            log(f"[{net} rep{rep}] DONE auprc={r['auprc']:.4f} base_rate={r['base_rate']:.4f} "
                f"auprc_x={r['auprc_x']:.3f}x prec@k={r['prec_at_k']:.3f} w={r['w']:.3f} "
                f"({r['runtime_sec']:.0f}s)\n")
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/twinscore_directed_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
