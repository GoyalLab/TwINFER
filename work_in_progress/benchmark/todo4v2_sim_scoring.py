"""
TODO4v2 scoring for the simulated benchmarks (network_sweep_final, e13_pos100,
mixed_network_sweep, real_networks) against their known ground-truth matrices.

RECONSTRUCTED 2026-09-20: the original file was lost from disk with no trace
(no git history -- work_in_progress/benchmark/ has never been tracked -- and
no filesystem snapshot available). Rebuilt from: (1) the intact LARRY-side
sibling `apply_todo4v2_allpairs_with_competitors.py`'s todo4v2_score (kept in
sync with this file per handoff/2026-09-18_todo4v2_signed_metrics_and_t1_10_pipeline.md
Part 2), (2) the exact nogate formula writeup in
handoff/2026-09-17_todo4v2_nogate_formula.md, (3) bytecode metadata recovered
from the surviving __pycache__/*.pyc (function signatures, docstrings, string/
numeric constants -- via xdis, since decompyle3 doesn't support 3.10+ bytecode),
and (4) direct inspection of the actual *_all_results.json / beeline_analysis_output.csv
schemas this session. The shared math (s, todo4v2_score's terms, _reg_and_flux)
is high-confidence; load_true_edges/score_one_json/_competitor_summary/the
four run_* functions are written fresh against the verified file formats, not
recovered verbatim -- no golden output CSV survived to regression-test against.
"""
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve

ROOT = "/home/gzu5140/TwINFER_KA"
HERE = "/home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark"
LARRY_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation"
sys.path.insert(0, LARRY_DIR)
from analytic_zscores import DEFAULT_SD, z_signed

P_VALUE = 0.01
Z_TWO_SIDED = float(norm.ppf(1 - P_VALUE / 2))   # ~2.576
Z_ONE_SIDED = float(norm.ppf(1 - P_VALUE))       # ~2.326
Z_HET_THR_NEW = -2.326

# real_networks: sim_type (as stored in each *_all_results.json) -> ground-truth
# connectivity-matrix filename under input_data/real_world_networks/
REAL_NETWORK_MATRIX = {
    "Circadian_cycle": "circadian.txt",
    "mCAD": "mCAD.txt",
    "VSC": "VSC.txt",
    "B_cell_activation": "B_cell.txt",
    "HSC": "HSC.txt",
    "EMT": "EMT.txt",
    "GSD": "GSD.txt",
    "Pluripotent": "Pluripotent.txt",
}


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


def score_one_json(json_path, gt_matrix_path_resolver):
    """twin_score_inputs only contains rows that survived TwINFER's own upstream significance
    filtering (step1 existence test etc) -- for these simulated-network runs (not the LARRY
    ALL_PAIRS=1/threshold=0 unrestricted mode), that can be a small fraction of the full
    n*(n-1) universe. Score over whatever pairs are present; report how many that is."""
    d = json.load(open(json_path))
    tsi = d["twin_score_inputs"]
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    if len(dd) == 0:
        return None

    z_reg_map = d.get("gated_regulation", {}).get("z_reg_gated", {})
    gene_names = d["gene_names"]
    dataset_id = d.get("dataset_id", d.get("analysis_key", os.path.basename(json_path)))

    matrix_path = gt_matrix_path_resolver(d)
    if not matrix_path or not os.path.exists(matrix_path):
        return None
    true_edges = load_true_edges(matrix_path, gene_names)

    U = list(zip(dd.gene_1, dd.gene_2))
    y = np.array([1 if p in true_edges else 0 for p in U])
    n_selected = len(U)
    n_universe = len(gene_names) * (len(gene_names) - 1)

    out = dict(dataset_id=dataset_id, n_selected=n_selected,
               n_universe=n_universe, n_true=int(y.sum()))
    if y.sum() < 1:
        return out

    for variant in ("old", "new", "nogate"):
        score, n_pass_gate = todo4v2_score(dd, U, z_reg_map, variant)
        m = full_report(score, y)
        prefix = f"todo4v2_{variant}"
        out[f"{prefix}_auprc_x"] = m["auprc_x"]
        out[f"{prefix}_topk"] = f"{m['tp']}/{m['k']} (of {n_selected} selected)"
        # k is set to n_true, so precision == recall at top-k by construction -> F1 == precision
        out[f"{prefix}_topk_f1"] = m["topk_precision"]
        out[f"{prefix}_pass_gate"] = n_pass_gate
        out[f"{prefix}_degenerate"] = m["degenerate"]
    return out


def _competitor_summary(beeline_csv, dataset_ids):
    """Mean directed 'auprc' per algorithm across every row (all datasets/reps/runs), and the
    single best-performing algorithm by that mean -- same 7 BEELINE methods used elsewhere."""
    if not os.path.exists(beeline_csv):
        return None, None
    df = pd.read_csv(beeline_csv)
    df = df[df.dataset_id.isin(dataset_ids)]
    if df.empty:
        return None, None
    m = df.groupby("algorithm")["auprc"].mean().sort_values(ascending=False)
    return m.index[0], float(m.iloc[0])


def _run_family(name, json_glob, gt_resolver, beeline_csv, out_csv):
    paths = sorted(glob(json_glob))
    print(f"{name}: {len(paths)} files")
    rows = [r for r in (score_one_json(p, gt_resolver) for p in paths) if r]
    if not rows:
        print(f"  no usable rows for {name}")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")
    means = {v: df[f"todo4v2_{v}_auprc_x"].mean() for v in ("old", "new", "nogate")}
    print(f"  mean todo4v2_old={means['old']:.3f}x  todo4v2_new={means['new']:.3f}x  "
          f"todo4v2_nogate={means['nogate']:.3f}x  (n={len(df)}/{len(paths)} usable)")
    if beeline_csv:
        best_name, best_val = _competitor_summary(beeline_csv, df.dataset_id.unique().tolist())
        if best_name:
            print(f"  best competitor (raw AUPRC, not x-random): {best_name}={best_val:.3f}")
    return df


def run_network_sweep_e13():
    _run_family(
        "network_sweep (e13_pos100)",
        f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs/*_all_results.json",
        lambda d: d.get("ground_truth_matrix"),
        f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output.csv",
        f"{HERE}/todo4v2_network_sweep_e13_results.csv",
    )


def run_network_sweep_final_broader():
    """The rest of network_sweep_final's OFAT topology sweep (e5/e9/e17 x pos0/pos50/pos100,
    150 replicates) -- e13_pos100 was scored separately by run_network_sweep_e13 above."""
    _run_family(
        "network_sweep_final (broader OFAT sweep)",
        f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs/*_all_results.json",
        lambda d: d.get("ground_truth_matrix"),
        f"{ROOT}/analysis_data/network_sweep_final/beeline_analysis_output.csv",
        f"{HERE}/todo4v2_network_sweep_final_broader_results.csv",
    )


def run_mixed_network_sweep():
    _run_family(
        "mixed_network_sweep",
        f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs/*_all_results.json",
        lambda d: f"{ROOT}/input_data/mixed_network_sweep/{d.get('dataset_id')}.txt",
        f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv",
        f"{HERE}/todo4v2_mixed_network_sweep_results.csv",
    )


def run_real_networks():
    """Ground truth: each JSON's own base_config path (not persisted directly in the current
    infer_real_network_allpairs.py record) -- reconstruct via the same NETWORKS/topology mapping
    that script uses, keyed off sim_type."""
    _run_family(
        "real_networks",
        f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/*_all_results.json",
        lambda d: f"{ROOT}/input_data/real_world_networks/{REAL_NETWORK_MATRIX.get(d.get('sim_type', ''), '')}",
        None,  # (no precomputed BEELINE competitor CSV found for real_networks -- reporting vs random only)
        f"{HERE}/todo4v2_real_networks_results.csv",
    )


if __name__ == "__main__":
    which = os.environ.get("WHICH", "all")
    runners = {
        "network_sweep_e13": run_network_sweep_e13,
        "network_sweep_final_broader": run_network_sweep_final_broader,
        "mixed_network_sweep": run_mixed_network_sweep,
        "real_networks": run_real_networks,
    }
    if which == "all":
        for fn in runners.values():
            fn()
    elif which in runners:
        runners[which]()
    else:
        raise ValueError(f"WHICH={which!r} not one of {list(runners)}")
