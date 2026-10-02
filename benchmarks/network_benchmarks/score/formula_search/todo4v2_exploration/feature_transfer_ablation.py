# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Single-feature signal strength across every topology in real_networks + network_sweep, to
identify which raw z-score terms are consistently informative (transferable) vs topology-
dependent/harmful. Also tests a few candidate TODO4v2 recombinations against the currently-
losing cases (mCAD, VSC, e9_pos0) and checks they don't break the winning ones.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import glob
import os
import sys
import re

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as m


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def raw_auprc_x(sc, y):
    sc = np.nan_to_num(np.asarray(sc, float), nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    rand = y.mean()
    if rand <= 0 or rand >= 1:
        return np.nan
    prec, rec, _ = precision_recall_curve(y, sc)
    return auc(rec, prec) / rand


def load_pairs(json_path, gt_resolver, ds_key="dataset_id"):
    d = json.load(open(json_path))
    tsi = d.get("twin_score_inputs")
    gr = d.get("gated_regulation")
    if tsi is None or gr is None:
        return None
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    if len(dd) < 5:
        return None
    gene_names = d["gene_names"]
    gt_path = gt_resolver(d)
    if gt_path is None or not os.path.exists(gt_path):
        return None
    true_edges = m.load_true_edges(gt_path, gene_names)
    U = list(zip(dd.gene_1, dd.gene_2))
    y = np.array([1 if p in true_edges else 0 for p in U])
    if y.sum() < 2 or y.sum() == len(y):
        return None
    z_reg_map = gr["z_reg_gated"]
    z_reg = np.array([m.lookup(z_reg_map, a, b) for a, b in U])
    return dd, U, y, z_reg, d.get(ds_key)


def features_for(dd, U, y, z_reg):
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy()
    z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_rho_change = dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_flux_old = m._reg_and_flux(dd, abs_valued=False)
    feats = dict(
        abs_rho_cross_xy=np.abs(dd.rho_cross_xy.to_numpy()),
        z_abs_t1=z_abs_t1, z_abs_t2=z_abs_t2, z_abs_change=z_abs_change,
        z_het=z_het, z_div=z_div, z_gamma=z_gamma, z_rho_change=z_rho_change,
        z_reg_gated=z_reg, z_dagger=z_dagger, z_flux=z_flux_old,
    )
    out = {}
    for name, v in feats.items():
        best = -np.inf
        for sign, tag in [(1, "+"), (-1, "-"), (None, "abs")]:
            vv = np.abs(v) if sign is None else sign * v
            x = raw_auprc_x(vv, y)
            if np.isfinite(x) and x > best:
                best = x
        out[name] = best
    return out


TOPOLOGIES = {
    "mCAD": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs',
             lambda d: f"{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/mCAD.txt", "sim_type"),
    "VSC": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs',
            lambda d: f"{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/VSC.txt", "sim_type"),
    "Circadian_cycle": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs',
                        lambda d: f"{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/circadian.txt", "sim_type"),
    "Pluripotent": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs',
                    lambda d: f"{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/Pluripotent.txt", "sim_type"),
}


def topo_group_ns(ds):
    match = re.match(r"grn_n6_(e\d+_pos\d+_\w+?)_rep\d+", ds)
    return match.group(1) if match else ds


all_results = {}

for topo_name, (jdir, gtr, dskey) in TOPOLOGIES.items():
    files = sorted(glob.glob(f"{jdir}/{topo_name}_rep_*_all_results.json"))
    rows = []
    for f in files:
        r = load_pairs(f, gtr, dskey)
        if r is None:
            continue
        dd, U, y, z_reg, ds = r
        if ds != topo_name:
            continue
        feats = features_for(dd, U, y, z_reg)
        rows.append(feats)
    if not rows:
        continue
    df = pd.DataFrame(rows)
    all_results[topo_name] = df.mean()
    print(f"[{topo_name}] n={len(rows)}")

# network_sweep e9_pos0
jdir = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs'
files = sorted(glob.glob(f"{jdir}/*_all_results.json"))
for target in ["e9_pos0_sign_ratio", "e5_pos100_density"]:
    rows = []
    for f in files:
        d0 = json.load(open(f))
        if topo_group_ns(d0.get("dataset_id", "")) != target:
            continue
        r = load_pairs(f, lambda d: d.get("ground_truth_matrix"), "dataset_id")
        if r is None:
            continue
        dd, U, y, z_reg, ds = r
        feats = features_for(dd, U, y, z_reg)
        rows.append(feats)
    if rows:
        df = pd.DataFrame(rows)
        all_results[target] = df.mean()
        print(f"[{target}] n={len(rows)}")

summary = pd.DataFrame(all_results).T
print()
print("=== single-feature auprc_x, mean per topology (rows=topology, cols=feature) ===")
print(summary.round(3).to_string())
summary.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score/formula_search/todo4v2_exploration/feature_transfer_summary.csv')
