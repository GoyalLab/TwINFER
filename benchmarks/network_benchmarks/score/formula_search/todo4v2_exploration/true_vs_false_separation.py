# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""For each feature and topology: mean(feature | true edge) vs mean(feature | false edge), and
the standardized separation (Cohen's d = (mean_true - mean_false) / pooled_std). This is
independent of the AUPRC/random ratio metric (which mixes in the random-baseline artifact) --
a direct measure of how much the feature actually separates true from false edges.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import glob
import re
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as m


def cohens_d(true_vals, false_vals):
    true_vals = true_vals[np.isfinite(true_vals)]
    false_vals = false_vals[np.isfinite(false_vals)]
    if len(true_vals) < 2 or len(false_vals) < 2:
        return np.nan
    n1, n2 = len(true_vals), len(false_vals)
    s1, s2 = true_vals.std(ddof=1), false_vals.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_sd <= 0 or not np.isfinite(pooled_sd):
        return np.nan
    return (true_vals.mean() - false_vals.mean()) / pooled_sd


def features_for(dd, best_abs=True):
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy()
    z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_rho_change = dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_flux = m._reg_and_flux(dd, abs_valued=False)
    return dict(
        abs_rho_cross_xy=np.abs(dd.rho_cross_xy.to_numpy()),
        z_abs_t1=z_abs_t1, z_abs_t2=z_abs_t2, z_abs_change=z_abs_change,
        z_het=np.abs(z_het), z_div=np.abs(z_div), z_gamma=np.abs(z_gamma),
        z_rho_change=np.abs(z_rho_change), z_dagger=np.abs(z_dagger), z_flux=np.abs(z_flux),
    )


def topo_group_ns(ds):
    match = re.match(r"grn_n6_(e\d+_pos\d+_\w+?)_rep\d+", ds)
    return match.group(1) if match else ds


TOPOLOGIES = {
    "mCAD": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs',
             lambda d: f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/mCAD.txt', "sim_type"),
    "VSC": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs',
            lambda d: f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/VSC.txt', "sim_type"),
    "Circadian_cycle": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs',
                        lambda d: f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/circadian.txt', "sim_type"),
    "Pluripotent": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs',
                    lambda d: f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/Pluripotent.txt', "sim_type"),
}

all_results = {}

for topo_name, (jdir, gtr, dskey) in TOPOLOGIES.items():
    files = sorted(glob.glob(f"{jdir}/{topo_name}_rep_*_all_results.json"))
    rows = []
    for f in files:
        d = json.load(open(f))
        if d.get(dskey) != topo_name:
            continue
        tsi = d.get("twin_score_inputs")
        if tsi is None:
            continue
        dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
        if len(dd) < 5:
            continue
        gene_names = d["gene_names"]
        true_edges = m.load_true_edges(gtr(d), gene_names)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in true_edges else 0 for p in U])
        if y.sum() < 2 or y.sum() == len(y):
            continue
        feats = features_for(dd)
        row = {}
        for name, v in feats.items():
            row[name] = cohens_d(v[y == 1], v[y == 0])
        rows.append(row)
    if rows:
        all_results[topo_name] = pd.DataFrame(rows).mean()
        print(f"[{topo_name}] n={len(rows)}")

jdir = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs'
files = sorted(glob.glob(f"{jdir}/*_all_results.json"))
for target in ["e9_pos0_sign_ratio", "e5_pos100_density"]:
    rows = []
    for f in files:
        d = json.load(open(f))
        if topo_group_ns(d.get("dataset_id", "")) != target:
            continue
        tsi = d.get("twin_score_inputs")
        if tsi is None:
            continue
        dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
        if len(dd) < 5:
            continue
        gene_names = d["gene_names"]
        true_edges = m.load_true_edges(d.get("ground_truth_matrix"), gene_names)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in true_edges else 0 for p in U])
        if y.sum() < 2 or y.sum() == len(y):
            continue
        feats = features_for(dd)
        row = {}
        for name, v in feats.items():
            row[name] = cohens_d(v[y == 1], v[y == 0])
        rows.append(row)
    if rows:
        all_results[target] = pd.DataFrame(rows).mean()
        print(f"[{target}] n={len(rows)}")

summary = pd.DataFrame(all_results).T
print()
print("=== Cohen's d (true edges vs false edges), by topology x feature ===")
print(summary.round(3).to_string())
summary.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score/formula_search/todo4v2_exploration/cohens_d_summary.csv')

meta = pd.DataFrame({
    "n_genes": [5, 8, 4, 36, 6, 6],
    "density": [0.650, 0.268, 0.583, 0.066, 0.300, 0.167],
}, index=["mCAD", "VSC", "Circadian_cycle", "Pluripotent", "e9_pos0_sign_ratio", "e5_pos100_density"])
merged = summary.join(meta)
print()
print("=== correlation of Cohen's d with n_genes ===")
print(merged.corr(numeric_only=True)["n_genes"].drop(["n_genes", "density"]).round(3).sort_values())
print()
print("=== correlation of Cohen's d with density ===")
print(merged.corr(numeric_only=True)["density"].drop(["n_genes", "density"]).round(3).sort_values())
