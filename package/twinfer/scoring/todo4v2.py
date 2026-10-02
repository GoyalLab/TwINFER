"""TODO4v2 scoring of a twin_score_inputs table against known ground truth (functions only; the file/glob drivers stay in benchmarks/network_benchmarks/score/todo4v2_sim_scoring.py).
[Moved on 2026-09-30. The original file was lost and reconstructed on 2026-09-20; the reconstruction was REGRESSION-TESTED on 2026-09-30 against the CSVs the original wrote:
mixed_network_sweep 144/144 rows, real_networks 65/65, e13_pos100 30/30 identical (tests/smoke/smoke_todo4v2_sim_scoring_vs_stored.py).]
s() here pushes non-finite values below the panel minimum (the analytic_core variant sets them to 0; calculate_twin_score uses ddof=0).
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve

from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

P_VALUE = 0.01
Z_TWO_SIDED = float(norm.ppf(1 - P_VALUE / 2))   # ~2.576
Z_ONE_SIDED = float(norm.ppf(1 - P_VALUE))       # ~2.326
Z_HET_THR_NEW = -2.326


def s(v):
    """Per-panel z-standardization; non-finite values pushed below the panel minimum."""
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def full_report(sc, y):
    """2026-09-17: a score vector with zero finite entries (every pair failed the gate) is a
    completely tied/degenerate ranking -- sklearn's precision_recall_curve on a constant array
    still returns a non-trivial-looking AUPRC number that depends only on class balance, not any
    real discrimination. Flag that case explicitly instead of reporting a misleading number."""
    sc = np.asarray(sc, float)
    degenerate = not np.isfinite(sc).any()
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0,
                        posinf=np.nanmax(sc[np.isfinite(sc)]) + 1.0 if np.isfinite(sc).any() else 1.0,
                        neginf=np.nanmin(sc[np.isfinite(sc)]) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    k = int(y.sum())
    order = np.argsort(-sc, kind="stable")
    top_k = order[:k]
    tp = int((y[top_k] == 1).sum())
    topk_prec = tp / k if k else float("nan")
    return dict(auprc=auprc, auprc_random=rand, auprc_x=auprc / rand if rand > 0 else float("nan"),
                topk_precision=topk_prec, tp=tp, k=k, degenerate=degenerate)


def lookup(z_reg_map, a, b):
    v = None
    if f"{a}__{b}" in z_reg_map:
        v = z_reg_map[f"{a}__{b}"]
    elif f"{b}__{a}" in z_reg_map:
        v = z_reg_map[f"{b}__{a}"]
    return np.nan if v is None else v


def _reg_and_flux(dd, abs_valued):
    """yscher's flux_dagger REG construction. abs_valued=False (the 'old'-variant REG): signed
    net directional flow. abs_valued=True (the 'new'-variant REG): REG(g) built from
    |z_rho_dagger| instead, measuring the MAGNITUDE of g's outward vs inward directional signal
    rather than yscher's signed net-flow version."""
    z_out_full = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_in_full = z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])
    if abs_valued:
        z_out_full = np.abs(z_out_full)
        z_in_full = np.abs(z_in_full)
    asym_full = dd.assign(z_out=z_out_full, z_in=z_in_full)
    regd = asym_full.groupby("gene_1")["z_out"].mean() - asym_full.groupby("gene_1")["z_in"].mean()
    REG = regd.to_dict()
    z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])
    return z_flux


def todo4v2_score(dd, U, z_reg_map, variant):
    """variant='old': signed REG, one-sided z_reg_gated>2.326 gate (matches the version that beat
    pidc on LARRY correlation_high/annotfilter). variant='new': abs REG, two-sided
    |z_reg_gated|>2.576 gate (the version that regressed across the LARRY panel). variant='nogate':
    old-style signed REG + new-style two-sided abs hinge_het, no gate at all -- the variant used
    throughout the simulated-benchmark comparisons (small networks gate out almost everything
    under old/new), per handoff/2026-09-17_todo4v2_nogate_formula.md.

    s_zdagger uses abs(z_dagger) in ALL THREE variants (2026-09-17 fix): z_dagger's magnitude is
    a strength-of-regulation/existence signal, its sign is a separate activation/repression call
    and shouldn't penalize true repressive edges as if they were absent.
    """
    if variant not in ("old", "new", "nogate"):
        raise ValueError("unknown variant " + str(variant))

    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    z_het = dd.z_het.to_numpy()
    z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    Cc = -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    s_zdagger = s(np.abs(z_dagger))

    if variant == "old":
        z_flux = _reg_and_flux(dd, abs_valued=False)
        hinge_het = np.where(z_het < -Z_ONE_SIDED, z_het, 0.0)
        gate = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & \
               (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)
    elif variant == "new":
        z_flux = _reg_and_flux(dd, abs_valued=True)
        hinge_het = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)
        gate = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & \
               (np.nan_to_num(np.abs(z_reg), nan=-np.inf) > Z_TWO_SIDED)
    else:  # nogate
        z_flux = _reg_and_flux(dd, abs_valued=False)
        hinge_het = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)
        gate = np.ones(len(dd), dtype=bool)

    score = z_abs_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger
    score = np.where(gate, score, -np.inf)
    return score, int(gate.sum())


def load_true_edges(matrix_path, gene_names):
    """Ground truth directly from a connectivity matrix .txt (M[i,j] = effect of gene i on gene
    j, per twinfer.simulation.gillespie_simulations' convention) -- the simulated benchmarks have
    exact known topology, unlike LARRY which needs an external database."""
    M = np.loadtxt(matrix_path, dtype=float, delimiter=",")
    if M.ndim == 0:
        M = M.reshape((1, 1))
    n = len(gene_names)
    if M.shape != (n, n):
        raise ValueError(f"{matrix_path}: shape {M.shape} != ({n}, {n})")
    edges = set()
    for i in range(n):
        for j in range(n):
            if M[i, j] != 0:
                edges.add((gene_names[i], gene_names[j]))
    return edges
