"""Apply TODO4v2 (both variants: old=signed-REG/one-sided-gate, new=abs-REG/two-sided-gate --
see paper_analysis/larry_hematopoiesis_validation/apply_todo4v2_allpairs_with_competitors.py)
to the network_sweep (e13_pos100) and mixed_network_sweep simulated-GRN benchmarks, whose
*_all_results.json files (from infer_with_twinfer) already carry the same twin_score_inputs +
gated_regulation.z_reg_gated schema LARRY does -- no new inference run needed.

Ground truth: network_sweep's JSON embeds ground_truth_matrix (a path to a CSV adjacency
matrix); mixed_network_sweep's ground truth is input_data/mixed_network_sweep/<dataset_id>.txt,
same CSV-adjacency-matrix format. Both use row=regulator, col=target, nonzero=edge.

Competitor comparison reuses the ALREADY-COMPUTED beeline_analysis_output.csv per benchmark
(7 methods: GENIE3/GRNBOOST2/PEARSON/PIDC/PPCOR/SCODE/SCSGL) rather than rerunning BEELINE.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve

HERE = "/home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark"
ROOT = "/home/gzu5140/TwINFER_KA"
import sys
sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from analytic_zscores import DEFAULT_SD, z_signed

P_VALUE = 0.01
Z_TWO_SIDED = float(norm.ppf(1 - P_VALUE / 2))
Z_ONE_SIDED = float(norm.ppf(1 - P_VALUE))
Z_HET_THR_NEW = -2.326


def s(v):
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
    real discrimination, which silently looked like a real result on network_sweep (93% of reps
    had pass_gate=0). Flag that case explicitly (degenerate=True, auprc/auprc_x=NaN) instead of
    reporting the artifact."""
    sc = np.asarray(sc, float)
    degenerate = not np.isfinite(sc).any()
    rand = float(y.mean())
    k = int(y.sum())
    if degenerate:
        return dict(auprc=float("nan"), auprc_random=rand, auprc_x=float("nan"),
                    topk_precision=float("nan"), topk_recall=float("nan"), topk_f1=float("nan"),
                    tp=0, k=k, n_selected=0, degenerate=True)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)

    # 2026-09-17: tie-aware top-k selection, matching the Aug-24 benchmark's EXACT convention
    # (top_k_tie_aware_selection_twinfer in real_networks_summary_table.py / the notebook it was
    # ported from): find the score value AT rank k, then select every pair with a score >= that
    # boundary (more than k pairs if several tie at the boundary). That function ALSO floors the
    # boundary up to the smallest strictly-positive score whenever the raw k-th value is a
    # "no evidence" placeholder tied with many other pairs -- for the cross-corr-only baseline
    # that placeholder is 0.0 (abs(rho_cross_xy) is never negative), so ties-at-zero could
    # otherwise swallow the whole universe. TODO4v2's placeholder for an unscored pair is -inf,
    # not 0.0 -- -inf can never tie with a real finite k-th-place value, so the analogous "floor
    # up to the smallest real value" guard is included here for literal consistency but is a
    # structural no-op (max(min_finite, boundary) == boundary whenever boundary is itself finite,
    # which it always is outside the already-separately-handled degenerate case).
    order = np.argsort(-sc, kind="stable")
    boundary = sc[order[k - 1]] if k > 0 else np.inf
    min_finite = sc[np.isfinite(sc)].min() if np.isfinite(sc).any() else -np.inf
    best_val = max(min_finite, boundary)
    selected = sc >= best_val
    n_selected = int(selected.sum())
    tp = int((y[selected] == 1).sum())
    precision = tp / n_selected if n_selected else 0.0
    recall = tp / k if k else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return dict(auprc=auprc, auprc_random=rand, auprc_x=auprc / rand if rand > 0 else float("nan"),
                topk_precision=precision, topk_recall=recall, topk_f1=f1,
                tp=tp, k=k, n_selected=n_selected, degenerate=False)


def lookup(z_reg_map, a, b):
    v = None
    if f"{a}__{b}" in z_reg_map:
        v = z_reg_map[f"{a}__{b}"]
    elif f"{b}__{a}" in z_reg_map:
        v = z_reg_map[f"{b}__{a}"]
    return np.nan if v is None else v


def _reg_and_flux(dd, abs_valued):
    z_out = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_in = z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])
    if abs_valued:
        z_out, z_in = np.abs(z_out), np.abs(z_in)
    asym = dd.assign(z_out=z_out, z_in=z_in)
    regd = asym.groupby("gene_1")["z_out"].mean() - asym.groupby("gene_1")["z_in"].mean()
    REG = regd.to_dict()
    return np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])


def todo4v2_score(dd, U, z_reg_map, variant):
    """variant='old': signed REG, one-sided z_reg_gated>2.326 gate (matches the version that beat
    pidc on LARRY correlation_high/annotfilter). variant='new': abs REG, two-sided
    |z_reg_gated|>2.576 gate (the version that regressed across the LARRY panel)."""
    abs_valued = (variant == "new")
    z_flux = _reg_and_flux(dd, abs_valued)
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    # 2026-09-17: existence gate now uses |z_abs_rho_t1|/|z_abs_rho_t2| instead of the raw
    # (possibly negative) values, per user instruction -- z_abs_rho is itself a z-score of a
    # |rho| statistic against its null mean, so a large-magnitude NEGATIVE value means the
    # observed |rho| is unusually SMALL (below what pure noise would produce), not a strong
    # existence signal; abs() here treats that the same as a strong positive signal.
    z_abs_t1_gate = np.abs(z_abs_t1)
    z_abs_t2_gate = np.abs(z_abs_t2)
    if variant == "old":
        gate = (np.maximum(z_abs_t1_gate, z_abs_t2_gate) > Z_TWO_SIDED) & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)
    elif variant == "new":
        gate = (np.maximum(z_abs_t1_gate, z_abs_t2_gate) > Z_TWO_SIDED) & (np.abs(np.nan_to_num(z_reg, nan=0.0)) > Z_TWO_SIDED)
    elif variant == "nogate":
        # 2026-09-17: both the existence gate AND the z_reg_gated gate removed, per user request
        # -- every pair gets a real score, none are floored to -inf. Diagnostic for how much of
        # TODO4v2's small-network degeneracy (network_sweep: 28/30 reps fully gated out) traces
        # to the gate itself vs. the underlying score terms.
        gate = np.ones(len(z_abs_t1), dtype=bool)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    Cc = -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)
    # 2026-09-17: z_dagger's score contribution is now abs-valued, per user instruction --
    # rho_cross_xy's magnitude is a strength-of-regulation signal (existence), its SIGN is a
    # separate activation-vs-repression call that shouldn't penalize true repressive edges by
    # scoring them as if they were absent. (Was s(z_dagger), signed -- that collapsed this
    # feature's Cohen's d as neg_frac rose, see scaling_analysis.py.)
    s_zdagger = s(np.abs(z_dagger))

    score = z_abs_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger
    score = np.where(gate, score, -np.inf)
    return score, int(gate.sum())


def load_true_edges(matrix_path, gene_names):
    M = np.loadtxt(matrix_path, delimiter=",")
    n = len(gene_names)
    assert M.shape == (n, n), f"{matrix_path}: shape {M.shape} != ({n},{n})"
    true = {(gene_names[i], gene_names[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    return true


def score_one_json(json_path, gt_matrix_path_resolver):
    """twin_score_inputs only contains rows that survived TwINFER's own upstream significance
    filtering (step1 existence test etc) -- for these simulated-network runs (not the LARRY
    ALL_PAIRS=1/threshold=0 unrestricted mode), that can be a small fraction of the full
    n*(n-1) universe. Score over the FULL universe (matching how the competitor AUPRC numbers
    in beeline_analysis_output.csv are computed): pairs absent from twin_score_inputs are
    treated as "never called" and floored to the bottom of the ranking, same convention as a
    gate-failing pair."""
    d = json.load(open(json_path))
    tsi = d.get("twin_score_inputs")
    gr = d.get("gated_regulation")
    if tsi is None or gr is None or gr.get("z_reg_gated") is None:
        return None
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    # 2026-09-17: pairs with NaN rho_cross_xy (zero-variance twin-paired subset at one timepoint
    # -- see grn_n10_e20_c8_a0_pos50_rep2_rep0's gene_10, near-silent at t1) are dropped from the
    # scored universe entirely (not inferred), matching the cross-corr-only baseline's convention
    # and per user instruction, rather than letting them silently pollute z_dagger/z_flux/REG's
    # panel standardization (s()'s NaN-handling previously pushed them to the min value instead
    # of excluding them, and REG(g) is a per-gene groupby mean that a single NaN row could skew).
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    if len(dd) < 3:
        return None
    z_reg_map = gr["z_reg_gated"]
    gene_names = d["gene_names"]
    gt_path = gt_matrix_path_resolver(d)
    if gt_path is None or not os.path.exists(gt_path):
        return None
    true_edges = load_true_edges(gt_path, gene_names)

    U_present = list(zip(dd.gene_1, dd.gene_2))
    scored_pairs = set(U_present)
    U_full = [(a, b) for a in gene_names for b in gene_names if a != b and (a, b) in scored_pairs]
    y_full = np.array([1 if p in true_edges else 0 for p in U_full])
    if y_full.sum() < 1 or y_full.sum() == len(y_full):
        return None

    out = dict(dataset_id=d.get("dataset_id", d.get("analysis_key", os.path.basename(json_path))),
               n_pairs=len(U_full), n_present=len(U_present), n_true=int(y_full.sum()))
    for variant in ("old", "new", "nogate"):
        sc_present, n_pass = todo4v2_score(dd, U_present, z_reg_map, variant)
        score_map = dict(zip(U_present, sc_present))
        floor = (np.nanmin(sc_present[np.isfinite(sc_present)]) - 1.0
                 if np.isfinite(sc_present).any() else -1.0)
        sc_full = np.array([score_map.get(p, floor) for p in U_full])
        rep = full_report(sc_full, y_full)
        out[f"todo4v2_{variant}_auprc_x"] = rep["auprc_x"]
        out[f"todo4v2_{variant}_topk"] = f"{rep['tp']}/{rep['k']} (of {rep['n_selected']} selected)"
        out[f"todo4v2_{variant}_topk_f1"] = rep["topk_f1"]
        out[f"todo4v2_{variant}_pass_gate"] = n_pass
        out[f"todo4v2_{variant}_degenerate"] = rep["degenerate"]
    return out


def _competitor_summary(beeline_csv, dataset_ids=None):
    """Mean directed 'auprc' per algorithm across every row (all datasets/reps/runs), and the
    single best-performing algorithm by that mean -- same 7 BEELINE methods used elsewhere."""
    if not os.path.exists(beeline_csv):
        return None, None
    b = pd.read_csv(beeline_csv)
    if dataset_ids is not None:
        b = b[b.dataset_id.isin(dataset_ids)]
    if b.empty:
        return None, None
    means = b.groupby("algorithm")["auprc"].mean().sort_values(ascending=False)
    return means.index[0], float(means.iloc[0])


def run_network_sweep_e13():
    JSON_DIR = f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs"
    BEELINE_CSV = f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output.csv"
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"network_sweep (e13_pos100): {len(files)} files")
    resolver = lambda d: d.get("ground_truth_matrix")
    rows = [r for r in (score_one_json(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/todo4v2_network_sweep_e13_results.csv", index=False)
    print(df.to_string(index=False))
    best_algo, best_auprc = _competitor_summary(BEELINE_CSV)
    print(f"\nmean todo4v2_old={df.todo4v2_old_auprc_x.mean():.3f}x  "
          f"todo4v2_new={df.todo4v2_new_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable)")
    if best_algo:
        print(f"best competitor (raw AUPRC, not x-random): {best_algo}={best_auprc:.3f}")
    return df


def run_network_sweep_final_broader():
    """The rest of network_sweep_final's OFAT topology sweep (e5/e9/e17 x pos0/pos50/pos100,
    150 replicates) -- e13_pos100 was scored separately by run_network_sweep_e13 above."""
    JSON_DIR = f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs"
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"network_sweep_final (broader OFAT sweep): {len(files)} files")
    resolver = lambda d: d.get("ground_truth_matrix")
    rows = [r for r in (score_one_json(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/todo4v2_network_sweep_final_broader_results.csv", index=False)
    print(f"\nmean todo4v2_old={df.todo4v2_old_auprc_x.mean():.3f}x  "
          f"todo4v2_new={df.todo4v2_new_auprc_x.mean():.3f}x  "
          f"todo4v2_nogate={df.todo4v2_nogate_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable)")
    return df


def run_mixed_network_sweep():
    JSON_DIR = f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs"
    GT_DIR = f"{ROOT}/input_data/mixed_network_sweep"
    BEELINE_CSV = f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv"
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"mixed_network_sweep: {len(files)} files")
    resolver = lambda d: f"{GT_DIR}/{d.get('dataset_id')}.txt"
    rows = [r for r in (score_one_json(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/todo4v2_mixed_network_sweep_results.csv", index=False)
    best_algo, best_auprc = _competitor_summary(BEELINE_CSV)
    print(f"\nmean todo4v2_old={df.todo4v2_old_auprc_x.mean():.3f}x  "
          f"todo4v2_new={df.todo4v2_new_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable)")
    if best_algo:
        print(f"best competitor (raw AUPRC, not x-random): {best_algo}={best_auprc:.3f}")
    return df


def run_real_networks():
    """Ground truth: each JSON's own base_config path (not persisted directly in the current
    infer_real_network_allpairs.py record) -- reconstruct via the same NETWORKS/topology mapping
    that script uses, keyed off sim_type."""
    JSON_DIR = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs"
    TOPO_DIR = f"{ROOT}/input_data/real_world_networks"
    TOPO_MAP = {
        "GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
        "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
        "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt",
    }
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"real_networks: {len(files)} files")
    resolver = lambda d: f"{TOPO_DIR}/{TOPO_MAP.get(d.get('sim_type'), '')}"
    rows = [r for r in (score_one_json(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/todo4v2_real_networks_results.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nmean todo4v2_old={df.todo4v2_old_auprc_x.mean():.3f}x  "
          f"todo4v2_new={df.todo4v2_new_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable)")
    print("(no precomputed BEELINE competitor CSV found for real_networks -- reporting vs random only)")
    return df


if __name__ == "__main__":
    which = os.environ.get("WHICH", "all")
    if which in ("network_sweep", "all"):
        run_network_sweep_e13()
    if which in ("network_sweep_final_broader", "all"):
        run_network_sweep_final_broader()
    if which in ("mixed_network_sweep", "all"):
        run_mixed_network_sweep()
    if which in ("real_networks", "all"):
        run_real_networks()
