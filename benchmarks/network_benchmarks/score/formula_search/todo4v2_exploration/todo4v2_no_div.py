# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""TODO4v2(nogate) with divp (z_div-based term) removed, z_gamma-based new_gamma kept, tested
against the same 6 topologies from the transferability ablation."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import glob
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as m

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


def score_no_div(dd, U, z_reg):
    """TODO4v2(nogate) minus divp (drops z_div entirely)."""
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_flux = m._reg_and_flux(dd, abs_valued=False)

    Cc = -s(z_abs_change)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)
    s_zdagger = s(z_dagger)

    # no gate, no divp
    return z_abs_t1 + Cc + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger


def raw_auprc_x(sc, y):
    sc = np.asarray(sc, float)
    if not np.isfinite(sc).any():
        return np.nan
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0)
    rand = y.mean()
    prec, rec, _ = precision_recall_curve(y, sc)
    return auc(rec, prec) / rand


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

results = {}
for topo_name, (jdir, gtr, dskey) in TOPOLOGIES.items():
    files = sorted(glob.glob(f"{jdir}/{topo_name}_rep_*_all_results.json"))
    xs = []
    for f in files:
        d = json.load(open(f))
        if d.get(dskey) != topo_name:
            continue
        tsi = d.get("twin_score_inputs")
        gr = d.get("gated_regulation")
        if tsi is None or gr is None:
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
        z_reg_map = gr["z_reg_gated"]
        z_reg = np.array([m.lookup(z_reg_map, a, b) for a, b in U])
        sc = score_no_div(dd, U, z_reg)
        xs.append(raw_auprc_x(sc, y))
    if xs:
        results[topo_name] = np.nanmean(xs)

jdir = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs'
files = sorted(glob.glob(f"{jdir}/*_all_results.json"))
for target in ["e9_pos0_sign_ratio", "e5_pos100_density"]:
    xs = []
    for f in files:
        d = json.load(open(f))
        if topo_group_ns(d.get("dataset_id", "")) != target:
            continue
        tsi = d.get("twin_score_inputs")
        gr = d.get("gated_regulation")
        if tsi is None or gr is None:
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
        z_reg_map = gr["z_reg_gated"]
        z_reg = np.array([m.lookup(z_reg_map, a, b) for a, b in U])
        sc = score_no_div(dd, U, z_reg)
        xs.append(raw_auprc_x(sc, y))
    if xs:
        results[target] = np.nanmean(xs)

print("=== TODO4v2(nogate) minus divp, auprc_x by topology ===")
for k, v in results.items():
    print(f"  {k:20s} {v:.3f}")
