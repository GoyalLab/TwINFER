#!/usr/bin/env python3
# HELPER LIBRARY (2026-09-23): shared functions (data loading, S/zhet, sister/cross correlation, PIDC, Wz, clr_calibrate,
# signal_share, s, null_sd, bootstrap_phi_noise_floor, ...) used by apply_twinscore_supplement_fatemap_gated_bootstrap.py.
# Copied verbatim from apply_twinscore_supplement_fatemap.py, which is now SUPERSEDED. The old persistence()/run_gene_set()/
# main() further down are NOT used by the final pipeline -- do not run this file directly.
"""Ports larry_hematopoiesis_validation/apply_twinscore_supplement_larry.py (TwinScore_supplement.pdf,
2026-09-17) to FM06/FM08/SpaceBar's single-timepoint data, at the user's explicit instruction to
run the FULL formula (phi/persistence + PAIR + direction_term) with t1=t2 rather than dropping the
cross-time terms -- correctly, t1=t2 does NOT make rho_dagger/phi trivial: _build_cross_time_twins
excludes only exact same-cell_id pairs, so rho_dagger(x,y) at t1=t2 is still a genuine within-clone,
cross-CELL (sibling) correlation, not a self-correlation. It measures "does gene x in one sibling
predict gene y in another sibling of the same clone" -- exactly the same question at one timepoint
that the manuscript asks across two.

TwinScore(x,y) = w_rel * s(phi_x) + s(PAIR) + direction_term
PAIR            = s(D) + gate_g * s(R) + gate_g * v_gate * s(Wz)
direction_term  = kappa_gamma * gate_dir * log(q)

Deliberate substitutions vs. the LARRY reference (each is dataset-specific instead of an
LARRY-calibrated borrowed constant -- same principle as apply_todo4v2_vs_collectri.py):
  - z_het, z_rho_t1/z_rho_t2: recomputed fresh via the package's own real-permutation functions
    (check_gene_gene_correlation_threshold, differentiate_single_state_reg_and_multiple_states)
    on THIS dataset, in place of yscher's precomputed LARRY-only JSON exports.
  - z_dagger (direction_term's cross-correlation z-score): reused directly from
    run_infer_fatemap.py's z_dagger_{label}_{geneset}.json -- the same empirical, dataset-specific
    quantity already computed by infer_with_twinfer for TODO4v2 -- instead of an analytic formula.
  - sd_reg (R's normalizer): null_sd(meff_step1) computed from THIS dataset's own clone-size
    distribution, in place of DEFAULT_SD["het_t1"] (a LARRY-calibrated constant).
  - CollecTRI: human (collectri_human.tsv), not mouse.
  - t1_raw and t2_raw are the SAME dataframe (single timepoint); S_t1==S_t2, z_rho_t1==z_rho_t2
    trivially, computed once and reused for both roles.
  - Twin construction uses the SAME oversized-clone cap (MAX_CLONE_SIZE=50) as run_infer_fatemap.py,
    for the same reason (FM08 barcode-collision artifact) -- applied uniformly, including to D's
    input cells, rather than reintroducing an unfiltered larger cell set for D alone.

PIDC (D) and Wz machinery are ported verbatim from the reference (dataset-agnostic).
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
import scipy.io as sio
from scipy.sparse import coo_matrix
from scipy.stats import norm
from scipy.stats import gamma as gamma_dist
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, split_twins, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
    get_unit_weights, weighted_spearman,
    differentiate_single_state_reg_and_multiple_states,
)

COLLECTRI_PATH = (
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
    f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
)
MAX_CLONE_SIZE = 50
SEED = 0
N_RAND_HET = 40
R0 = 0.69
Z_DIRECTION_GATE = 2.576
Z_ONE_SIDED = 2.326


def log(msg):
    print(msg, flush=True)


# ============================================================== shared helpers (verbatim ports)
# [2026-10-01 KNOWN ISSUE, label only, not fixed (user: this module is unlikely to be used later): the m_eff_* functions here are pair-weighted
#   (Kish without the multiplicity factor) and differ from twinfer.scoring.analytic_zscores (clone-weighted Kish). Neither is exact. Measured vs permutation SD:
#   Step-1 rho: clone-weighted -5%, pair-weighted +13%; sister rho (FM06): clone-weighted +41%, pair-weighted -21%. Rank-based scores are unaffected;
#   saved meff / z-score diagnostics from this module are only approximate. Kept as separate functions on purpose. See REVIEW_LOG.md 'm_eff'.
from twinfer.scoring.twinscore_supplement import (s, signal_share, clr_calibrate, null_sd, m_eff_step1, m_eff_twin, m_eff_cross)  # noqa: F401
# [2026-09-30 moved to twinfer.scoring (twinscore_supplement): s commented out here]
# def s(v):
#     v = np.asarray(v, float)
#     f = np.isfinite(v)
#     o = np.zeros(len(v))
#     if f.sum() > 1:
#         o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
#     o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
#     return o


# [2026-09-30 moved to twinfer.scoring (twinscore_supplement): signal_share commented out here]
# def signal_share(z):
#     z = np.asarray(z, float)
#     z = z[np.isfinite(z)]
#     if len(z) < 2:
#         return 0.0
#     return max(0.0, 1.0 - 1.0 / z.var(ddof=1))


# [2026-09-30 moved to twinfer.scoring (twinscore_supplement): clr_calibrate commented out here]
# def clr_calibrate(M, genes):
#     row_mean, row_sd = {}, {}
#     for g in genes:
#         vals = np.array([M.loc[g, h] for h in genes if h != g and np.isfinite(M.loc[g, h])])
#         row_mean[g] = vals.mean() if len(vals) else 0.0
#         row_sd[g] = vals.std(ddof=1) if len(vals) > 1 else 1.0
#
#     def c(x, y):
#         mxy = M.loc[x, y]
#         if not np.isfinite(mxy):
#             return np.nan
#         ux = (mxy - row_mean[x]) / max(row_sd[x], 1e-12)
#         uy = (mxy - row_mean[y]) / max(row_sd[y], 1e-12)
#         return np.sqrt(max(0.0, ux) ** 2 + max(0.0, uy) ** 2)
#     return c


# [2026-09-30 moved to twinfer.scoring (twinscore_supplement): null_sd commented out here]
# def null_sd(meff):
#     return 1.0 / np.sqrt(max(meff - 1.0, 1e-9))


# [2026-09-30 moved to twinfer.scoring (twinscore_supplement): m_eff_step1 commented out here]
# def m_eff_step1(clone_sizes):
#     n = np.asarray(clone_sizes, float)
#     w = 1.0 / n
#     return float(w.sum() ** 2 / (w ** 2 * n).sum()) if len(n) else np.nan


# [2026-09-30 moved to twinfer.scoring (twinscore_supplement): m_eff_twin commented out here]
# def m_eff_twin(clone_sizes):
#     n = np.asarray(clone_sizes, float)
#     npairs = n * (n - 1) / 2.0
#     return float(npairs.sum() ** 2 / (npairs ** 2 / npairs.clip(min=1)).sum()) if len(n) else np.nan


# [2026-09-30 moved to twinfer.scoring (twinscore_supplement): m_eff_cross commented out here]
# def m_eff_cross(n1, n2):
#     n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
#     w = n1 * n2
#     return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else np.nan


# ============================================================== data loading (FM06/FM08 adaptation)
def load_raw(dataset, gene_set_name):
    label = dataset.lower()
    qc_dir = f"{TWINFER_PROJECT_ROOT}/finalized_data/{dataset}_data/qc_filtered"
    gene_sets = json.load(open(f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/gene_sets_{label}.json"))
    genes = sorted(gene_sets[gene_set_name])

    X = sio.mmread(f"{qc_dir}/{label}_qc_counts.mtx").tocsr()
    genes_full = np.array(open(f"{qc_dir}/genes.txt").read().split())
    obs = pd.read_csv(f"{qc_dir}/obs_metadata.csv", index_col=0)
    assert X.shape[0] == len(obs)
    gidx = {g: i for i, g in enumerate(genes_full)}
    col = [gidx[g] for g in genes]
    cell_total = np.asarray(X.sum(axis=1)).ravel().astype(float)
    mat = np.log1p(X[:, col].toarray().astype(float) / np.where(cell_total == 0, 1, cell_total)[:, None] * 1e4)

    df = pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in genes])
    df.insert(0, "time_step", 0)
    df.insert(0, "cell_id", obs.index.to_numpy())
    clone = obs["fatemap_clone_singletcode"].astype(str)
    df.insert(0, "clone_id", clone.to_numpy())

    has_clone = clone.str.len() > 0
    df = df[has_clone.to_numpy()].reset_index(drop=True)
    counts = df["clone_id"].value_counts()
    keep = counts[(counts >= 2) & (counts <= MAX_CLONE_SIZE)].index
    dropped = int((~df["clone_id"].isin(keep)).sum())
    df = df[df["clone_id"].isin(keep)].reset_index(drop=True)
    log(f"[{dataset}/{gene_set_name}] {len(df):,} cells, {df['clone_id'].nunique():,} clones "
        f"(dropped {dropped:,} cells: no clone, singleton, or >MAX_CLONE_SIZE={MAX_CLONE_SIZE})")
    return df, genes, qc_dir, genes_full, gidx


# ============================================================== S, C/h, z_het (fresh, real)
def same_cell_matrix(raw, genes):
    return calculate_pairwise_gene_gene_correlation_matrix(raw, genes, use_clone=True)


def sister_matrix_from_frame(raw, genes):
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
    return C, m_eff_twin(clone_sizes)


def compute_zhet_zrho(raw, genes, n_shuffles=2000, n_cores=8):
    twin_delta, random_delta = calculate_twin_random_correlations(
        raw, assign_twin_id(raw), genes, random_state=SEED, unit="clone",
    )
    (multiple_states_pairs, single_state_regulation, _, z_het_scores, _) = \
        differentiate_single_state_reg_and_multiple_states(
            raw, [(a, b) for i, a in enumerate(genes) for b in genes[i + 1:]],
            twin_delta, random_delta, genes,
            z_score_threshold=5, verbose=False, unit="clone",
            n_shuffles=n_shuffles, n_cores_to_use=n_cores,
            divergence_random_state=SEED + 271832, return_divergence_details=True,
        )
    zhet_matrix = pd.DataFrame(np.nan, index=genes, columns=genes)
    for (a, b), z in z_het_scores.items():
        if z is not None:
            zhet_matrix.loc[a, b] = zhet_matrix.loc[b, a] = float(z)
    corr_t = same_cell_matrix(raw, genes)
    return zhet_matrix, corr_t, twin_delta


# ============================================================== cross-sample (within-clone,
# cross-cell) correlation: persistence numerator
def cross_matrix(t1_raw, t2_raw, genes):
    at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
    ordered = [(a, b) for a in genes for b in genes if a != b]
    xc = get_cross_correlations(at1, at2, gene_pairs=ordered, unit="clone")
    both = set(t1_raw.clone_id) & set(t2_raw.clone_id)
    n2 = t1_raw[t1_raw.clone_id.isin(both)].groupby("clone_id").size()
    n4 = t2_raw[t2_raw.clone_id.isin(both)].groupby("clone_id").size()
    both = sorted(both)
    meff = m_eff_cross(n2.loc[both].to_numpy(), n4.loc[both].to_numpy())
    return xc, meff


# ============================================================== persistence phi_g, reliability w
def bootstrap_phi_noise_floor(t1_raw, t2_raw, genes, thr1, thr2, n_boot=20, seed=SEED):
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
    return phi, w


# ============================================================== D: real PIDC (verbatim port)
from twinfer.scoring.pidc import (bayesian_blocks, discretise, joint, mutual_information, specific_information, pidc_scores)  # noqa: F401
# [2026-09-30 moved to twinfer.scoring (pidc): bayesian_blocks commented out here]
# def bayesian_blocks(x, p0=0.05):
#     x = np.sort(np.asarray(x, dtype=float))
#     n = len(x)
#     if n < 3 or np.all(x == x[0]):
#         return np.array([x[0] - 0.5, x[0] + 0.5])
#     edges = np.concatenate([x[:1], 0.5 * (x[1:] + x[:-1]), x[-1:]])
#     block_length = x[-1] - edges
#     best = np.zeros(n, dtype=float)
#     last = np.zeros(n, dtype=int)
#     ncp_prior = 4 - np.log(73.53 * p0 * n ** -0.478)
#     for k in range(n):
#         width = block_length[:k + 1] - block_length[k + 1]
#         count = np.arange(k + 1, 0, -1)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             fit = count * (np.log(count) - np.log(np.where(width > 0, width, np.nan)))
#         fit = np.nan_to_num(fit, nan=-np.inf, neginf=-np.inf)
#         A = fit - ncp_prior
#         A[1:] += best[:k]
#         i = int(np.argmax(A))
#         last[k] = i
#         best[k] = A[i]
#     ch = []
#     i = n
#     while i > 0:
#         ch.append(i)
#         i = last[i - 1]
#     return edges[np.sort(np.array(ch + [0]))]


# [2026-09-30 moved to twinfer.scoring (pidc): discretise commented out here]
# def discretise(X, how="bayesian_blocks"):
#     out = np.zeros_like(X, dtype=int)
#     nb = []
#     for j in range(X.shape[1]):
#         col = X[:, j]
#         e = bayesian_blocks(col) if how == "bayesian_blocks" else np.histogram_bin_edges(col, bins=10)
#         e = np.unique(e)
#         if len(e) < 2:
#             e = np.array([col.min() - .5, col.max() + .5])
#         out[:, j] = np.clip(np.digitize(col, e[1:-1]), 0, len(e) - 2)
#         nb.append(len(e) - 1)
#     return out, np.array(nb)


# [2026-09-30 moved to twinfer.scoring (pidc): joint commented out here]
# def joint(a, b, na, nb):
#     t = np.zeros((na, nb))
#     np.add.at(t, (a, b), 1.0)
#     return t / t.sum()


# [2026-09-30 moved to twinfer.scoring (pidc): mutual_information commented out here]
# def mutual_information(P):
#     px = P.sum(1, keepdims=True)
#     py = P.sum(0, keepdims=True)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         v = P * (np.log(P) - np.log(px) - np.log(py))
#     return float(np.nansum(np.where(P > 0, v, 0.0)))


# [2026-09-30 moved to twinfer.scoring (pidc): specific_information commented out here]
# def specific_information(P):
#     pz = P.sum(0)
#     px = P.sum(1)
#     out = np.zeros(P.shape[1])
#     for zi in range(P.shape[1]):
#         if pz[zi] <= 0:
#             continue
#         pxz = P[:, zi] / pz[zi]
#         with np.errstate(divide="ignore", invalid="ignore"):
#             pzx = np.where(px > 0, P[:, zi] / px, 0.0)
#             term = np.where(pzx > 0, np.log(1.0 / pz[zi]) - np.log(1.0 / pzx), 0.0)
#         out[zi] = float(np.nansum(pxz * term))
#     return out, pz


# [2026-09-30 moved to twinfer.scoring (pidc): pidc_scores commented out here]
# def pidc_scores(X, genes, how="bayesian_blocks"):
#     B, nb = discretise(X, how)
#     g = len(genes)
#     log(f"    PIDC discretisation: bins per gene min {nb.min()} median {int(np.median(nb))} max {nb.max()}")
#     MI = np.zeros((g, g))
#     JP = {}
#     for i in range(g):
#         for j in range(i + 1, g):
#             P = joint(B[:, i], B[:, j], nb[i], nb[j])
#             JP[(i, j)] = P
#             MI[i, j] = MI[j, i] = mutual_information(P)
#     SI = {}
#     for i in range(g):
#         for j in range(g):
#             if i == j:
#                 continue
#             P = JP[(i, j)] if i < j else JP[(j, i)].T
#             SI[(i, j)] = specific_information(P)
#     PUC = np.zeros((g, g))
#     for i in range(g):
#         for j in range(g):
#             if i == j:
#                 continue
#             mij = MI[i, j]
#             if mij <= 0:
#                 continue
#             tot = 0.0
#             for k in range(g):
#                 if k == i or k == j:
#                     continue
#                 si_i, pz = SI[(i, j)]
#                 si_k, _ = SI[(k, j)]
#                 red = float(np.sum(pz * np.minimum(si_i, si_k)))
#                 tot += (mij - red) / mij
#             PUC[i, j] = tot
#     D = np.zeros((g, g))
#     for i in range(g):
#         v = PUC[i, np.arange(g) != i]
#         v = v[np.isfinite(v)]
#         try:
#             a, loc, sc = gamma_dist.fit(v[v > 0], floc=0)
#             F = lambda u, a=a, sc=sc: gamma_dist.cdf(u, a, loc=0, scale=sc)
#         except Exception:
#             mu, sd = float(np.mean(v)), float(np.std(v) or 1.0)
#             F = lambda u, mu=mu, sd=sd: norm.cdf(u, mu, sd)
#         for j in range(g):
#             if i != j:
#                 D[i, j] += F(PUC[i, j])
#     return pd.DataFrame(D, index=genes, columns=genes)


def compute_D(raw, genes):
    X = raw[[f"{g}_mRNA" for g in genes]].to_numpy(dtype=float)
    return pidc_scores(X, genes)


# ============================================================== Wz: twin-layer partial correlation
def clone_weights_from_ids(clone_ids):
    counts = pd.Series(clone_ids).value_counts()
    return (1.0 / pd.Series(clone_ids).map(counts)).to_numpy()


def pooled_matrix(x_a, x_b, y_a, y_b, w):
    w2 = np.concatenate([w, w]) / 2.0
    px = np.concatenate([x_a, x_b])
    py = np.concatenate([y_b, y_a])
    return weighted_spearman(px, py, w2)


def _one_split_half_W(X4, clone_ids, genes, genes_full, seed):
    rng = np.random.default_rng(seed)
    X4c = X4.tocoo()
    half1_data = rng.binomial(X4c.data.astype(np.int64), 0.5).astype(float)
    half2_data = X4c.data - half1_data
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


def compute_Wz(raw, genes, qc_dir, label, genes_full, gidx, seed=SEED, n_boot=20):
    t0 = time.time()
    X_full = sio.mmread(f"{qc_dir}/{label}_qc_counts.mtx").tocsr()
    obs = pd.read_csv(f"{qc_dir}/obs_metadata.csv", index_col=0)
    kept_ids = set(raw["cell_id"])
    mask = obs.index.isin(kept_ids)
    X4 = X_full[mask]
    obs4 = obs.loc[mask]
    order = pd.Series(range(len(raw)), index=raw["cell_id"]).reindex(obs4.index)
    keep2 = order.notna()
    X4 = X4[keep2.to_numpy()]
    clone_ids = raw.set_index("cell_id").loc[obs4.index[keep2.to_numpy()], "clone_id"].to_numpy()

    n = len(genes)
    W_draws, meff_twin_list = [], []
    for b in range(n_boot):
        Wb, meff_twin_b = _one_split_half_W(X4, clone_ids, genes, genes_full, seed=seed + b)
        W_draws.append(Wb)
        meff_twin_list.append(meff_twin_b)
    W_draws = np.stack(W_draws, axis=0)
    meff_twin_ = float(np.mean(meff_twin_list))
    meff_T = m_eff_step1(pd.Series(clone_ids).value_counts().to_numpy())
    log(f"    Wz: {n_boot} bootstrap split-half draws done ({time.time()-t0:.0f}s), {X4.shape[0]:,} cells")

    W_mean = W_draws.mean(axis=0)
    W_var_boot = W_draws.var(axis=0, ddof=1)
    iu = np.triu_indices(n, k=1)
    denom = np.sum(W_mean[iu] ** 2)
    a_star = float(np.clip(W_var_boot[iu].sum() / denom, 0.0, 1.0)) if denom > 0 else 1.0

    def shrink_invert(Wm):
        Wshrunk = (1 - a_star) * Wm + a_star * np.eye(n)
        P = -np.linalg.pinv(Wshrunk + 1e-8 * np.eye(n))
        d = np.sqrt(np.clip(np.outer(np.diag(P), np.diag(P)), 1e-12, None))
        PC = P / d
        np.fill_diagonal(PC, 0.0)
        return PC

    PC_mean = shrink_invert(W_mean)
    PC_draws = np.stack([shrink_invert(W_draws[b]) for b in range(n_boot)], axis=0)
    se = PC_draws.std(axis=0, ddof=1)
    # 2026-09-22 fix: with only n_boot=20 draws, se occasionally lands within noise of exactly
    # 0 by chance even where PC_mean itself is unremarkable -- dividing by that near-zero se
    # produced individual Wz values in the millions (checked directly: 22/1122 pairs on
    # FM06/correlation_high, up to 1.3e7), and because s() standardizes over the WHOLE panel,
    # those few outliers corrupted s(Wz) for every pair in the gene set, not just themselves.
    # A floor scaled to this panel's own typical |PC_mean| (not the previous fixed 1e-9,
    # which is far below any real partial-correlation magnitude) keeps se from vanishing
    # relative to the quantity it's dividing, and a hard clip on the resulting z is a second,
    # independent guard against any single-draw outlier still slipping through.
    iu_full = ~np.eye(n, dtype=bool)
    se_floor = max(1e-9, 0.05 * float(np.median(np.abs(PC_mean[iu_full]))))
    se_safe = np.where(se > se_floor, se, se_floor)
    Wz_raw = PC_mean / se_safe
    Wz_clipped = np.clip(Wz_raw, -20.0, 20.0)
    n_clipped = int(np.sum((Wz_raw != Wz_clipped) & iu_full))
    if n_clipped:
        log(f"    Wz: clipped {n_clipped} outlier z-value(s) to +/-20 (se_floor={se_floor:.4g})")
    Wz = pd.DataFrame(Wz_clipped, index=genes, columns=genes)
    v = signal_share(Wz.to_numpy()[~np.eye(n, dtype=bool)])
    log(f"    Wz: a*={a_star:.3f}  v={v:.3f}  (meff_T={meff_T:.0f}, meff_twin={meff_twin_:.0f})")
    return Wz, v, a_star


# ============================================================== main per-gene-set pipeline
def run_gene_set(dataset, gs, ce_edges, n_shuffles=2000, n_cores=8, compute_wz=True):
    label = dataset.lower()
    raw, genes, qc_dir, genes_full, gidx = load_raw(dataset, gs)
    t1_raw = t2_raw = raw  # single timepoint: t1 IS t2, same underlying cells

    log(f"[{dataset}/{gs}] real S, z_het (fresh permutation, this dataset)")
    zhet_matrix, S, twin_delta = compute_zhet_zrho(raw, genes, n_shuffles=n_shuffles, n_cores=n_cores)
    S_t1 = S_t2 = S

    log(f"[{dataset}/{gs}] C, heritability (fresh weighted-Spearman)")
    C_t1, meff_twin_t1 = sister_matrix_from_frame(t1_raw, genes)
    C_t2, meff_twin_t2 = C_t1, meff_twin_t1  # same cells
    h_t1 = {g: C_t1.loc[g, g] for g in genes}
    h_t2 = h_t1

    log(f"[{dataset}/{gs}] cross-cell (within-clone sibling) correlation -- persistence numerator")
    rho_dagger, meff_cross_ = cross_matrix(t1_raw, t2_raw, genes)
    rho_dagger_diag = {g: rho_dagger.loc[g, g] for g in genes}

    log(f"[{dataset}/{gs}] persistence phi_g, reliability w")
    phi, w_rel = persistence(h_t1, h_t2, rho_dagger_diag, genes, meff_twin_t1, meff_twin_t2, t1_raw, t2_raw)

    log(f"[{dataset}/{gs}] zreg, R, twin-signal gate g")
    n_genes = len(genes)
    meff_step1 = m_eff_step1(raw.groupby("clone_id").size().to_numpy())
    sd_reg = null_sd(meff_step1)  # dataset-specific, NOT LARRY's DEFAULT_SD["het_t1"]
    sd_twin_t2 = null_sd(meff_twin_t2)
    lam = np.minimum(1.0, zhet_matrix.abs() / 2.33)
    zreg = (S_t2 - lam * C_t2) / sd_reg
    R_fn = clr_calibrate(zreg, genes)
    zC = C_t2 / sd_twin_t2
    gate_g = signal_share(zC.to_numpy()[~np.eye(n_genes, dtype=bool)])

    if compute_wz:
        log(f"[{dataset}/{gs}] D (real PIDC) and Wz (split-half partial correlation)")
    else:
        log(f"[{dataset}/{gs}] D (real PIDC)")
    D_mat = compute_D(t2_raw, genes)
    if compute_wz:
        Wz_mat, v_gate, a_star = compute_Wz(raw, genes, qc_dir, label, genes_full, gidx)
    else:
        Wz_mat = pd.DataFrame(0.0, index=genes, columns=genes)
        v_gate, a_star = 0.0, np.nan

    log(f"[{dataset}/{gs}] direction: z^dagger (reused from run_infer_fatemap.py), gamma, kappa_gamma, q")
    z_dagger_path = (f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/"
                      f"z_dagger_{label}_{gs}.json")
    z_dagger_map = json.load(open(z_dagger_path)) if os.path.exists(z_dagger_path) else {}
    U = [(a, b) for a in genes for b in genes if a != b]

    def zdag(a, b):
        return z_dagger_map.get(f"{a}__{b}", np.nan)

    z_dag_xy = np.array([zdag(a, b) for a, b in U])
    z_dag_yx = np.array([zdag(b, a) for a, b in U])
    gamma_raw = (np.abs(z_dag_xy) - np.abs(z_dag_yx)) / np.sqrt(2 * (1 - 2 / np.pi))
    kappa_gamma = signal_share(gamma_raw)
    q = 0.5 + (R0 - 0.5) * (2 * norm.cdf(np.abs(gamma_raw)) - 1) * np.sign(gamma_raw)
    q = np.clip(np.nan_to_num(q, nan=0.5), 1e-6, 1 - 1e-6)
    gate_dir = (np.nan_to_num(np.maximum(np.abs(z_dag_xy), np.abs(z_dag_yx)), nan=0.0)
                > Z_DIRECTION_GATE).astype(float)
    direction_term = kappa_gamma * gate_dir * np.log(q)

    log(f"[{dataset}/{gs}] assembling PAIR and TwinScore")
    D_vec = np.array([D_mat.loc[a, b] for a, b in U])
    R_vec = np.array([R_fn(a, b) for a, b in U])
    Wz_vec = np.array([Wz_mat.loc[a, b] for a, b in U])
    PAIR = s(D_vec) + gate_g * s(R_vec) + gate_g * v_gate * s(Wz_vec)
    phi_x = np.array([phi[a] for a, b in U])
    score = w_rel * s(phi_x) + s(PAIR) + direction_term

    pair_df = pd.DataFrame({
        "gene_1": [a for a, b in U], "gene_2": [b for a, b in U],
        "S": [S_t2.loc[a, b] for a, b in U], "C": [C_t2.loc[a, b] for a, b in U],
        "zhet": [zhet_matrix.loc[a, b] for a, b in U], "lambda": [lam.loc[a, b] for a, b in U],
        "zreg": [zreg.loc[a, b] for a, b in U], "R": R_vec, "gate_g": gate_g,
        "D": D_vec, "Wz": Wz_vec, "gate_v": v_gate,
        "rho_dagger_xy": [rho_dagger.loc[a, b] for a, b in U],
        "rho_dagger_yx": [rho_dagger.loc[b, a] for a, b in U],
        "z_dagger_xy": z_dag_xy, "z_dagger_yx": z_dag_yx, "gamma": gamma_raw, "kappa_gamma": kappa_gamma,
        "q": q, "direction_term": direction_term, "PAIR": PAIR, "TwinScore": score,
        "collectri_edge": [1 if p in ce_edges else 0 for p in U],
    })
    gene_df = pd.DataFrame({
        "gene": genes, "h": [h_t2[g] for g in genes],
        "rho_dagger_gg": [rho_dagger_diag[g] for g in genes], "phi": [phi[g] for g in genes],
    })
    diagnostics = dict(dataset=dataset, gs=gs, n_genes=n_genes, n_pairs=len(U),
                        meff_twin=meff_twin_t2, meff_cross=meff_cross_, meff_step1=meff_step1,
                        w=w_rel, gate_g=gate_g, gate_v=v_gate, kappa_gamma=kappa_gamma,
                        wz_shrinkage_a=a_star, median_h=float(np.median(list(h_t2.values()))))
    return pair_df, gene_df, diagnostics


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
                auroc=auroc, prec=tp / k if k else float("nan"), tp=tp, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["FM06", "FM08"])
    ap.add_argument("--gene-set", default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=2000)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--no-wz", action="store_true")
    args = ap.parse_args()

    ct = pd.read_csv(COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    label = args.dataset.lower()
    out_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
    os.makedirs(out_dir, exist_ok=True)

    pair_df, gene_df, diag = run_gene_set(
        args.dataset, args.gene_set, CE, n_shuffles=args.n_shuffles,
        n_cores=args.n_cores, compute_wz=not args.no_wz,
    )
    pair_df.to_csv(os.path.join(out_dir, f"twinscore_supplement_{label}_{args.gene_set}_pair_terms.csv"), index=False)
    gene_df.to_csv(os.path.join(out_dir, f"twinscore_supplement_{label}_{args.gene_set}_gene_terms.csv"), index=False)

    y = pair_df.collectri_edge.to_numpy()
    rep = full_report(pair_df.TwinScore.to_numpy(), y)
    pair_rep = full_report(pair_df.PAIR.to_numpy(), y)
    log(f"[{args.dataset}/{args.gene_set}] n_pairs={len(pair_df)} n_true={int(y.sum())} "
        f"TwinScore={rep['auprc_x']:.3f}x  PAIR-alone={pair_rep['auprc_x']:.3f}x  "
        f"w={diag['w']:.3f} g={diag['gate_g']:.3f} v={diag['gate_v']:.3f} "
        f"kappa_gamma={diag['kappa_gamma']:.3f}")

    summary_path = f'{TWINFER_PROJECT_ROOT}/analysis_data/fatemap_twinscore_supplement_summary.csv'
    row = dict(dataset=args.dataset, gene_set=args.gene_set, n_pairs=len(pair_df), n_true=int(y.sum()),
               auprc_x=rep["auprc_x"], pair_alone_auprc_x=pair_rep["auprc_x"],
               auroc=rep["auroc"],
               **{k: v for k, v in diag.items() if k not in ("dataset", "gs", "n_pairs")})
    row_df = pd.DataFrame([row])
    if os.path.exists(summary_path):
        prior = pd.read_csv(summary_path)
        prior = prior[~((prior.dataset == args.dataset) & (prior.gene_set == args.gene_set))]
        row_df = pd.concat([prior, row_df], ignore_index=True)
    row_df.to_csv(summary_path, index=False)
    log(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
