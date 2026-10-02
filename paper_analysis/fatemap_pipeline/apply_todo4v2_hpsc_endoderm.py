#!/usr/bin/env python3
"""TODO4v2 scoring on hPSC_20260927, ported from apply_todo4v2_vs_collectri.py (FM06/FM08).
Same deliberate substitution as that port: z_dagger is the EMPIRICAL direction-stage z-score
infer_with_twinfer actually computed for this dataset (direction_z_scores.csv, from the real
n_shuffles permutation null), not the package todo4v2.py's analytic z_signed(...,
DEFAULT_SD["cross"]) -- that constant is calibrated to LARRY's clone-size structure and would
silently be wrong here.

Ground truth: the 592-edge ChIP-Atlas+Perturb-seq candidate set (NOT CollecTRI -- user
explicitly rejected CollecTRI as ground truth for this dataset once real ChIP-seq+Perturb-seq
data existed).

Scoring convention (2026-10-01, fixed to match the rest of this session): FULL gene-pair
UNIVERSE for each panel (not just the pairs that happen to appear in ranked_edges.csv), with
any unscored/missing pair filled at the worst observed score for that method -- the same
convention used everywhere else in this pipeline, after an earlier version of this script was
caught scoring only the narrower "ranked_edges" subset (a pre-filter artifact). Reports
TODO4v2, twinScore, and all 4 competitors (rho/ppcor/pidc/grnboost2) side by side per run.

Gate matches apply_todo4v2_vs_collectri.py exactly: max(z_abs_t1,z_abs_t2) > 2.576 AND
z_reg_gated > 2.326 (one-sided, "old"-variant gate).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: replaces hardcoded project-root paths]
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# DATA_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_20260927/data"   # [2026-10-01 replaced by TWINFER_PROJECT_ROOT]
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
GT_PATH = f"{DATA_DIR}/hpsc_endoderm_candidate_pairs_final.csv"

# run -> (full_results dir, panel json, networks gene-set label)
RUNS = {
    "top100": ("top100_full_results", "hpsc_endoderm_top100_panel.json", "top100"),
    "gt50": ("gt50_full_results", "hpsc_endoderm_gt50_panel.json", "gt50"),
    "gt50_nofilter": ("gt50_nofilter_full_results", "hpsc_endoderm_gt50_panel.json", "gt50"),
}
COMPETITORS = ["rho", "ppcor", "pidc", "grnboost2"]

Z_HET_THR_NEW = -2.326
Z_TWO_SIDED = 2.576
Z_ONE_SIDED = 2.326


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def lookup(d, a, b, default=np.nan):
    if f"{a}__{b}" in d:
        return d[f"{a}__{b}"]
    if f"{b}__{a}" in d:
        return d[f"{b}__{a}"]
    return default


def todo4v2_score(dd, z_reg_map, z_dagger_map):
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy()
    z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()

    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in zip(dd.gene_1, dd.gene_2)])
    z_dagger = np.array([lookup(z_dagger_map, a, b) for a, b in zip(dd.gene_1, dd.gene_2)])

    tmp = dd.assign(_zd=np.abs(z_dagger))
    regd = tmp.groupby("gene_1")["_zd"].mean() - tmp.groupby("gene_2")["_zd"].mean()
    REG = regd.to_dict()
    z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])

    gate = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)

    Cc = -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(z_het < -Z_ONE_SIDED, z_het, 0.0)
    s_zdagger = s(np.abs(z_dagger))

    score = z_abs_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger
    score = np.where(gate, score, -np.inf)
    return score, int(gate.sum())


def full_universe_score(df, g1, g2, score_col, universe):
    """Best (max) score per unordered pair over the WHOLE panel universe; pairs missing from
    df (never scored) get the worst observed score, same convention used elsewhere this
    session -- NaN/inf scores are dropped before taking the max/min."""
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col]).copy()
    d["ukey"] = [frozenset((a, b)) for a, b in zip(d[g1], d[g2])]
    best = d.groupby("ukey")[score_col].max()
    worst = best.min() - 1.0 if len(best) else -1.0
    return pd.Series([best.get(u, worst) for u in universe], index=universe)


def auprc_x(scores, gt_set, universe, base_rate):
    y = np.array([1 if u in gt_set else 0 for u in universe])
    ap = average_precision_score(y, scores.to_numpy())
    return ap, ap / base_rate if base_rate > 0 else float("nan")


def main():
    gt = pd.read_csv(GT_PATH)
    GT = set(frozenset(p) for p in zip(gt.TF, gt.target))
    print(f"ground truth (ChIP-Atlas+Perturb-seq, undirected): {len(GT)} edges", flush=True)

    rows = []
    for run, (out_subdir, panel_json, net_label) in RUNS.items():
        out_dir = f"{DATA_DIR}/{out_subdir}"
        re_path = f"{out_dir}/ranked_edges.csv"
        zreg_path = f"{out_dir}/gated_regulation_z_reg_gated.csv"
        zdag_path = f"{out_dir}/direction_z_scores.csv"
        if not all(os.path.exists(p) for p in (re_path, zreg_path, zdag_path)):
            print(f"[{run}] SKIP -- missing outputs", flush=True)
            continue

        panel = json.load(open(f"{DATA_DIR}/{panel_json}"))["panel"]
        universe = [frozenset(p) for p in itertools.combinations(panel, 2)]
        gt_in_panel = GT & set(universe)
        base_rate = len(gt_in_panel) / len(universe)
        print(f"\n[{run}] panel={len(panel)} genes, universe={len(universe)} pairs, "
              f"ground truth in panel={len(gt_in_panel)} (base rate {base_rate:.4f})", flush=True)
        if len(gt_in_panel) < 3:
            print(f"[{run}] SKIP scoring -- too few true edges in this panel for a meaningful AUPRC", flush=True)
            continue

        dd = pd.read_csv(re_path)
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        zreg_df = pd.read_csv(zreg_path)
        z_reg_map = {f"{a}__{b}": v for a, b, v in zip(zreg_df.gene_1, zreg_df.gene_2, zreg_df.value)}
        zdag_df = pd.read_csv(zdag_path)
        z_dagger_map = {f"{a}__{b}": v for a, b, v in zip(zdag_df.gene_1, zdag_df.gene_2, zdag_df.value)}

        score, n_pass_gate = todo4v2_score(dd, z_reg_map, z_dagger_map)
        dd["todo4v2"] = score

        row = dict(run=run, n_panel_genes=len(panel), n_universe_pairs=len(universe),
                   n_true=len(gt_in_panel), n_pass_gate=n_pass_gate)

        s_todo4v2 = full_universe_score(dd, "gene_1", "gene_2", "todo4v2", universe)
        ap, apx = auprc_x(s_todo4v2, gt_in_panel, universe, base_rate)
        row["todo4v2_auprc_x"] = apx

        if "twinScore" in dd.columns:
            s_tw = full_universe_score(dd, "gene_1", "gene_2", "twinScore", universe)
            ap, apx = auprc_x(s_tw, gt_in_panel, universe, base_rate)
            row["twinscore_auprc_x"] = apx

        for method in COMPETITORS:
            net_path = f"{DATA_DIR}/networks/{method}_{net_label}_allgenes.csv"
            if not os.path.exists(net_path):
                row[f"{method}_auprc_x"] = np.nan
                continue
            cdf = pd.read_csv(net_path)
            c = cdf.columns.tolist()
            s_c = full_universe_score(cdf, c[0], c[1], c[2], universe)
            ap, apx = auprc_x(s_c, gt_in_panel, universe, base_rate)
            row[f"{method}_auprc_x"] = apx

        rows.append(row)
        print(f"[{run}] AUPRC-x -- todo4v2={row['todo4v2_auprc_x']:.3f}  "
              f"twinscore={row.get('twinscore_auprc_x', float('nan')):.3f}  "
              + "  ".join(f"{m}={row[f'{m}_auprc_x']:.3f}" for m in COMPETITORS), flush=True)

    df = pd.DataFrame(rows)
    out_csv = f"{DATA_DIR}/hpsc_endoderm_todo4v2_vs_chipatlas.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}", flush=True)
    print("\n" + df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
