# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""User-specified new TwinScore formula, tested on the 6 topologies already loaded (mCAD, VSC,
Circadian_cycle, Pluripotent, e9_pos0_sign_ratio, e5_pos100_density).

TwinScore(x->y) = max_t s(c_t(x,y)) + s(z_flux(x->y))
                  - I[gate_stable] * s(z_stable)
                  + I[m < -2.326] * s(m)
Filters: (|z_rho(t1)|>2.576 or |z_rho(t2)|>2.576) and (z_reg_gated(t1)>2.326 or z_reg_gated(t2)>2.326)

c_t(x,y) = sqrt(max(0,u_x(y))^2 + max(0,u_y(x))^2)
u_x(y) = (z(x,y) - mean_w z(x,w)) / sd_w z(x,w),  z = z_abs_rho_t (per timepoint)
z_flux(x,y) = reg(x) - reg(y), reg(g) = mean_w z_rho_dagger(g->w) - mean_w z_rho_dagger(w->g)
z_stable = z_abs_rho_change  (z(|rho(t2)-rho(t1)|), already computed by the package)
m = z_het  (APPROXIMATION: package only computes z_het at t1, no native z_het_t2, so
    max(z_het(t1), z_het(t2)) -> z_het(t1) alone)

Only ONE gate for z_reg_gated is available (t1) in our schema, same limitation as z_het --
"z_reg_gated(t1) or z_reg_gated(t2)" collapses to just z_reg_gated(t1).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import glob
import os
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


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def per_gene_stats(dd, z_col):
    """mu[g], sd[g]: mean/sd of z_col(g, w) over all w, with g as SOURCE (gene_1)."""
    grp = dd.groupby("gene_1")[z_col]
    mu = grp.mean()
    sd = grp.std(ddof=1)
    sd = sd.where((sd > 0) & np.isfinite(sd), 1e-12)
    return mu, sd


def u_for(dd, z_col, which_gene):
    """u_x(y) or u_y(x) for every row (x,y): (z(x,y) - mu[g]) / sd[g], g = x if which_gene=='gene_1'
    else g = y -- both using the SAME z(x,y) value, only the reference gene's own row-stats differ."""
    mu, sd = per_gene_stats(dd, z_col)
    z = dd[z_col].to_numpy()
    ref_gene = dd[which_gene]
    mu_g = ref_gene.map(mu).to_numpy()
    sd_g = ref_gene.map(sd).to_numpy()
    return (z - mu_g) / sd_g


def new_formula_score(dd, U, z_reg):
    gene_names = sorted(set(dd.gene_1) | set(dd.gene_2))
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy()  # approximation: t1 only

    # u_x(y) = (z(x,y) - mean_w z(x,w)) / sd_w z(x,w) -- x's own row stats, x = gene_1
    ux_t1 = u_for(dd, "z_abs_rho_t1", "gene_1")
    ux_t2 = u_for(dd, "z_abs_rho_t2", "gene_1")
    # u_y(x) = (z(x,y) - mean_w z(y,w)) / sd_w z(y,w) -- y's own row stats (y as SOURCE, i.e.
    # y's stats come from rows where y is gene_1), applied to the SAME z(x,y) value
    uy_t1 = u_for(dd, "z_abs_rho_t1", "gene_2")
    uy_t2 = u_for(dd, "z_abs_rho_t2", "gene_2")

    c_t1 = np.sqrt(np.maximum(0, ux_t1) ** 2 + np.maximum(0, uy_t1) ** 2)
    c_t2 = np.sqrt(np.maximum(0, ux_t2) ** 2 + np.maximum(0, uy_t2) ** 2)
    existence = np.maximum(s(c_t1), s(c_t2))

    z_flux = m._reg_and_flux(dd, abs_valued=False)

    z_stable = z_abs_change
    gate_stable = (z_stable > Z_ONE_SIDED) & (np.abs(z_abs_t1) > Z_TWO_SIDED) & (np.abs(z_abs_t2) > Z_TWO_SIDED)
    hinge_stable = np.where(gate_stable, s(z_stable), 0.0)

    het_hinge = np.where(z_het < -Z_ONE_SIDED, s(z_het), 0.0)

    score = existence + s(z_flux) - hinge_stable + het_hinge

    gate = (np.abs(z_abs_t1) > Z_TWO_SIDED) | (np.abs(z_abs_t2) > Z_TWO_SIDED)
    gate = gate & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)
    score = np.where(gate, score, -np.inf)
    return score, int(gate.sum())


def raw_auprc_x(sc, y):
    sc = np.asarray(sc, float)
    if not np.isfinite(sc).any():
        return np.nan, True
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0)
    rand = y.mean()
    prec, rec, _ = precision_recall_curve(y, sc)
    return auc(rec, prec) / rand, False


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
        gt_path = gtr(d)
        true_edges = m.load_true_edges(gt_path, gene_names)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in true_edges else 0 for p in U])
        if y.sum() < 2 or y.sum() == len(y):
            continue
        z_reg_map = gr["z_reg_gated"]
        z_reg = np.array([m.lookup(z_reg_map, a, b) for a, b in U])
        sc, n_pass = new_formula_score(dd, U, z_reg)
        x, degenerate = raw_auprc_x(sc, y)
        xs.append((x, degenerate, n_pass))
    if xs:
        vals = [x for x, deg, n in xs if not deg]
        results[topo_name] = dict(n=len(xs), n_valid=len(vals),
                                   mean_auprc_x=np.mean(vals) if vals else np.nan)

# network_sweep topologies
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
        gt_path = d.get("ground_truth_matrix")
        true_edges = m.load_true_edges(gt_path, gene_names)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in true_edges else 0 for p in U])
        if y.sum() < 2 or y.sum() == len(y):
            continue
        z_reg_map = gr["z_reg_gated"]
        z_reg = np.array([m.lookup(z_reg_map, a, b) for a, b in U])
        sc, n_pass = new_formula_score(dd, U, z_reg)
        x, degenerate = raw_auprc_x(sc, y)
        xs.append((x, degenerate, n_pass))
    if xs:
        vals = [x for x, deg, n in xs if not deg]
        results[target] = dict(n=len(xs), n_valid=len(vals),
                                mean_auprc_x=np.mean(vals) if vals else np.nan)

print("=== new formula auprc_x by topology ===")
for k, v in results.items():
    print(f"  {k:20s} n={v['n']:3d} valid={v['n_valid']:3d}  mean_auprc_x={v['mean_auprc_x']:.3f}")
