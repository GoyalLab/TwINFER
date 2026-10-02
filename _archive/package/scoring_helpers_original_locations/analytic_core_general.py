"""Generalized (arbitrary gene count) version of the from-scratch analytic pipeline used
throughout this session (network_sweep_final's full_table_signed), parameterized by GENES so
it can run on network_sweep_final (6 genes), mixed_network_sweep (10 genes), and the real_data
sims of real_world_networks (variable gene count, e.g. GSD=19). Also provides the full updated
score: existence(|rho_t1|,|rho_t2|) + direction(gamma) + z_fanout + 0.5*z_d_het, with the SAME
fixed weights tuned on the LARRY gene sets (w_dir=1.0, w_fan=1.0, w_dhet=0.5) -- applied
out-of-sample here, not re-fit per dataset.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys

import numpy as np
import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from paper_analysis.larry_hematopoiesis_validation.analytic_zscores import (
    m_effs_from_table, analytic_twin_score_inputs, m_eff_step1, m_eff_twin, m_eff_cross, m_eff_d,
)

N_RAND = 20
SEED = 0

W_DIR, W_FAN, W_DHET = 1.0, 1.0, 0.5


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    return o


def full_table_disjoint(csv_path, T1, T2, GENES, seed=0, usecols=None):
    """Disjoint 1:1:2 clone-partition design, EXACTLY infer.py's simulation-data split
    (infer_with_twinfer, lines ~420-464 and ~548-553) -- not a reimplementation from the
    figure_4 reference script, the actual production split code:

      clone_ids shuffled -> quarters: t1_clones (1), t2_clones (1), across_t_clones (2).
      t1_twins_raw  = clones in t1_clones, time==T1, BOTH replicates (real twin pairs)
      t2_twins_raw  = clones in t2_clones, time==T2, BOTH replicates
      across_t_left_raw  = clones in across_t_clones, time==T1, replicate==replicates[0] only
      across_t_right_raw = clones in across_t_clones, time==T2, replicate==replicates[1] only
      across_t_twin1/2   = _build_cross_time_twins(across_t_left_raw, across_t_right_raw)
      rho_t1_measurements = concat(t1_twins_raw, across_t_left_raw)   -- plain rho_t1 pool
      rho_t2_measurements = concat(t2_twins_raw, across_t_right_raw)  -- plain rho_t2 pool

    Twin-difference stats (z_het, z_div, the random-pair reference) use ONLY the dedicated
    quarter's twin pairs (t1_twins/t2_twins) as the twin_table -- the number of random-pair
    draws always equals the number of real twin pairs, never more. The larger
    rho_t1_measurements/rho_t2_measurements pool is passed as raw_cells (where to draw each
    random partner FROM), exactly matching infer.py's own call pattern -- it does not change
    how many random pairs are drawn.

    No clone contributes to more than one of {t1 stats, t2 stats, cross-time stats}.
    """
    gcols = [f"{g}_mRNA" for g in GENES]
    cols = usecols or (["clone_id", "cell_id", "time_step", "replicate"] + gcols)
    df = pd.read_csv(csv_path, usecols=cols)
    df = df[df.time_step.isin([T1, T2])].reset_index(drop=True)

    replicates = np.sort(df["replicate"].drop_duplicates().to_numpy())
    if len(replicates) < 2:
        return None

    rng = np.random.default_rng(seed)
    clone_ids = df["clone_id"].drop_duplicates().to_numpy()
    clone_ids_shuffled = rng.permutation(clone_ids)
    n1 = n2 = len(clone_ids_shuffled) // 4
    t1_clones = clone_ids_shuffled[:n1]
    t2_clones = clone_ids_shuffled[n1:n1 + n2]
    across_t_clones = clone_ids_shuffled[n1 + n2:]

    t1_twins_raw = df[df["clone_id"].isin(t1_clones) & (df["time_step"] == T1)].copy()
    t2_twins_raw = df[df["clone_id"].isin(t2_clones) & (df["time_step"] == T2)].copy()
    across_t_left_raw = df[df["clone_id"].isin(across_t_clones) & (df["time_step"] == T1)
                           & (df["replicate"] == replicates[0])].copy()
    across_t_right_raw = df[df["clone_id"].isin(across_t_clones) & (df["time_step"] == T2)
                            & (df["replicate"] == replicates[1])].copy()

    if len(t1_twins_raw) < 20 or len(t2_twins_raw) < 20 or len(across_t_left_raw) < 20 or len(across_t_right_raw) < 20:
        return None

    rho_t1_measurements = pd.concat([t1_twins_raw, across_t_left_raw], ignore_index=True)
    rho_t2_measurements = pd.concat([t2_twins_raw, across_t_right_raw], ignore_index=True)

    t1_twins = assign_twin_id(t1_twins_raw).reset_index(drop=True)
    t2_twins = assign_twin_id(t2_twins_raw).reset_index(drop=True)
    across_t_twin1, across_t_twin2 = _build_cross_time_twins(across_t_left_raw, across_t_right_raw)

    g1 = t1_twins_raw.groupby("clone_id").size().to_numpy()
    g2 = t2_twins_raw.groupby("clone_id").size().to_numpy()
    sd1 = 1.0 / np.sqrt(max(m_eff_step1(g1) - 1, 1e-6))
    sd2 = 1.0 / np.sqrt(max(m_eff_step1(g2) - 1, 1e-6))
    div1 = 1.0 / np.sqrt(max(m_eff_twin(g1) - 1, 1e-6))
    div2 = 1.0 / np.sqrt(max(m_eff_twin(g2) - 1, 1e-6))

    # m_eff_cross needs clone sizes AT EACH TIMEPOINT for clones present at both -- across_t_left_raw
    # (T1, replicate[0] only) and across_t_right_raw (T2, replicate[1] only) each give exactly one
    # cell per across-time clone, so size is always 1 there; do NOT route this through
    # m_effs_from_table (its internal twin_t1/twin_t2 sub-calc needs n>=2 twin pairs, which this
    # single-replicate-per-side pool never has, and silently NaNs).
    gl = across_t_left_raw.groupby("clone_id").size()
    gr = across_t_right_raw.groupby("clone_id").size()
    both = sorted(set(gl.index) & set(gr.index))
    m_cross = m_eff_cross([gl[c] for c in both], [gr[c] for c in both])
    m_d = m_eff_d(m_eff_twin(g1), m_eff_twin(g2))  # disjoint t1/t2 twin quarters -> variances add

    SD = dict(step1_t1=sd1, step1_t2=sd2, div_t1=div1, het_t1=div1,
              d=1.0 / np.sqrt(max(m_d - 1, 1e-6)),
              change=float(np.sqrt(sd1 ** 2 + sd2 ** 2)),
              cross=1.0 / np.sqrt(max(m_cross - 1, 1e-6)))

    rho = {}
    rho["rho_t1"] = calculate_pairwise_gene_gene_correlation_matrix(rho_t1_measurements, GENES, use_clone=True)
    rho["rho_t2"] = calculate_pairwise_gene_gene_correlation_matrix(rho_t2_measurements, GENES, use_clone=True)
    rho["rho_delta_t1"], _ = calculate_twin_random_correlations(rho_t1_measurements, t1_twins, GENES, random_state=seed, unit="clone")
    rho["rho_delta_t2"], _ = calculate_twin_random_correlations(rho_t2_measurements, t2_twins, GENES, random_state=seed, unit="clone")
    r1 = np.mean([calculate_twin_random_correlations(rho_t1_measurements, t1_twins, GENES, random_state=seed + 100 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    r2 = np.mean([calculate_twin_random_correlations(rho_t2_measurements, t2_twins, GENES, random_state=seed + 500 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    rho["rho_delta_random_t1"] = pd.DataFrame(r1, index=GENES, columns=GENES)
    rho["rho_delta_random_t2"] = pd.DataFrame(r2, index=GENES, columns=GENES)

    ordered = [(a, b) for a in GENES for b in GENES if a != b]
    rho["rho_cross"] = get_cross_correlations(across_t_twin1, across_t_twin2,
                                               gene_pairs=ordered + [(g, g) for g in GENES], unit="clone")

    tsi = analytic_twin_score_inputs(rho, SD=SD).set_index(["gene_1", "gene_2"])
    return tsi


def full_table_signed_general(csv_path, T1, T2, GENES, usecols=None):
    gcols = [f"{g}_mRNA" for g in GENES]
    cols = usecols or (["clone_id", "cell_id", "time_step"] + gcols)
    df = pd.read_csv(csv_path, usecols=cols)
    t1_raw = df[df.time_step == T1].reset_index(drop=True)
    t2_raw = df[df.time_step == T2].reset_index(drop=True)
    if len(t1_raw) < 20 or len(t2_raw) < 20:
        return None
    t1_tw = assign_twin_id(t1_raw).reset_index(drop=True)
    t2_tw = assign_twin_id(t2_raw).reset_index(drop=True)

    M = m_effs_from_table(df, t1=T1, t2=T2)
    sd1 = 1.0 / np.sqrt(max(M["step1_t1"] - 1, 1e-6))
    sd2 = 1.0 / np.sqrt(max(M["step1_t2"] - 1, 1e-6))
    SD = dict(step1_t1=sd1, step1_t2=sd2,
              div_t1=1.0 / np.sqrt(max(M["twin_t1"] - 1, 1e-6)),
              het_t1=1.0 / np.sqrt(max(M["twin_t1"] - 1, 1e-6)),
              d=1.0 / np.sqrt(max(M["d"] - 1, 1e-6)),
              change=float(np.sqrt(sd1 ** 2 + sd2 ** 2)),
              cross=1.0 / np.sqrt(max(M["cross"] - 1, 1e-6)))

    rho = {}
    rho["rho_t1"] = calculate_pairwise_gene_gene_correlation_matrix(t1_raw, GENES, use_clone=True)
    rho["rho_t2"] = calculate_pairwise_gene_gene_correlation_matrix(t2_raw, GENES, use_clone=True)
    rho["rho_delta_t1"], _ = calculate_twin_random_correlations(t1_raw, t1_tw, GENES, random_state=SEED, unit="clone")
    rho["rho_delta_t2"], _ = calculate_twin_random_correlations(t2_raw, t2_tw, GENES, random_state=SEED, unit="clone")
    r1 = np.mean([calculate_twin_random_correlations(t1_raw, t1_tw, GENES, random_state=SEED + 100 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    r2 = np.mean([calculate_twin_random_correlations(t2_raw, t2_tw, GENES, random_state=SEED + 500 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    rho["rho_delta_random_t1"] = pd.DataFrame(r1, index=GENES, columns=GENES)
    rho["rho_delta_random_t2"] = pd.DataFrame(r2, index=GENES, columns=GENES)
    at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
    ordered = [(a, b) for a in GENES for b in GENES if a != b]
    rho["rho_cross"] = get_cross_correlations(at1, at2, gene_pairs=ordered + [(g, g) for g in GENES], unit="clone")

    tsi = analytic_twin_score_inputs(rho, SD=SD).set_index(["gene_1", "gene_2"])
    return tsi


def compute_scores(tsi, GENES):
    """Returns dict pair -> dict(existdir=..., full=...) using the fixed LARRY-tuned weights."""
    ordered = [(a, b) for a in GENES for b in GENES if a != b]
    tab = tsi.reindex(ordered)

    exist = s(tab.rho_t1.abs().to_numpy()) + s(tab.rho_t2.abs().to_numpy())
    rev = {(b, a): abs(v) for (a, b), v in zip(ordered, tab.rho_cross_xy.to_numpy())}
    direction = np.array([abs(tab.loc[p].rho_cross_xy) - rev[p] for p in ordered])
    s_dir = s(direction)

    n = len(GENES); gx = {g: i for i, g in enumerate(GENES)}
    Z = np.zeros((n, n))
    for (a, b), v in zip(ordered, tab.z_abs_rho_t1.to_numpy()):
        Z[gx[a], gx[b]] = v
    Z = np.maximum(Z, Z.T)
    np.fill_diagonal(Z, -np.inf)
    zfan = np.zeros(len(ordered))
    for k, (a, b) in enumerate(ordered):
        i, j = gx[a], gx[b]
        ci, cj = Z[i, :].copy(), Z[j, :].copy()
        ci[j] = -np.inf; cj[i] = -np.inf
        zfan[k] = np.max(np.minimum(ci, cj))
    s_fan = s(zfan)
    s_dhet = s(tab.z_d_het.to_numpy())

    existdir = exist + W_DIR * s_dir
    full = exist + W_DIR * s_dir + W_FAN * s_fan + W_DHET * s_dhet
    return {p: dict(existdir=existdir[i], full=full[i]) for i, p in enumerate(ordered)}


def score_topk(mag, true, poss):
    from sklearn.metrics import auc, precision_recall_curve
    y = np.array([1 if p in true else 0 for p in poss], int)
    k = int(y.sum())
    if k == 0 or k == len(poss):
        return dict(auprc=np.nan, f1=np.nan, precision=np.nan, recall=np.nan)
    x = np.array([mag[p] for p in poss], float)
    prec, rec, _ = precision_recall_curve(y, x)
    auprc = auc(rec, prec)
    order = np.argsort(-x, kind="stable")
    boundary = x[order[k - 1]]
    sel = np.where(x >= boundary)[0]
    tp = y[sel].sum()
    precision = tp / len(sel); recall = tp / k
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return dict(auprc=auprc, f1=f1, precision=precision, recall=recall)
