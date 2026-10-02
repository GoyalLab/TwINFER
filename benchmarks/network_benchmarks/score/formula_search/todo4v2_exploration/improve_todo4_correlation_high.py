# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Test concrete candidate TODO4 variants for correlation_high, building on the earlier
leave-one-out ablation: z_abs_t2 hurt (+0.109/+0.122 to drop), hinge_dagger mildly hurt
(+0.009/+0.025 to drop), hinge_het/hinge_stable/s_zflux near-neutral, z_abs_t1 helps (hurt to
drop), z_reg_gated-only gate ~= full dual gate, |z_rho|-only gate clearly worse.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources'
R = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
HERE = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score/formula_search/todo4v2_exploration'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

GS = "correlation_high"
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
    return auprc / rand if rand > 0 else float("nan"), tp / k, tp, k


def lookup(z_reg_map, a, b):
    v = None
    if f"{a}__{b}" in z_reg_map:
        v = z_reg_map[f"{a}__{b}"]
    elif f"{b}__{a}" in z_reg_map:
        v = z_reg_map[f"{b}__{a}"]
    return np.nan if v is None else v


def run(twin_def, analytic_dir, infer_suffix):
    dd = pd.read_csv(f"{R}/{analytic_dir}/{GS}/twin_score_inputs.csv")
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    z_reg_map = json.load(open(f"{R}/infer_results/{GS}_t2_t4_allpairs{infer_suffix}/z_scores_by_step.json"))["z_reg_gated"]

    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    U = list(zip(dd.gene_1, dd.gene_2))
    y = np.array([1 if p in CE else 0 for p in U])

    # yscher-exact REG (as already implemented in the scripts)
    z_dagger_out_full = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_dagger_in_full = z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])
    asym_full = dd.assign(z_out=z_dagger_out_full, z_in=z_dagger_in_full)
    regd = asym_full.groupby("gene_1")["z_out"].mean() - asym_full.groupby("gene_1")["z_in"].mean()
    REG = regd.to_dict()
    z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])

    z_abs_t1 = dd.z_abs_rho_t1.to_numpy(); z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    z_het = dd.z_het.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    gate_full = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)
    gate_reg_only = np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED

    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)
    hinge_dagger = -np.where(np.abs(z_dagger) > Z_ONE_SIDED, np.abs(z_dagger), 0.0)
    s_zflux = s(z_flux)

    print(f"\n{'='*70}\n{twin_def}  n_pairs={len(U)} n_true={y.sum()}\n{'='*70}")

    variants = {}
    variants["TODO4 (current, full gate)"] = (gate_full, z_abs_t1 + z_abs_t2 + s_zflux + hinge_stable + hinge_het + hinge_dagger)
    variants["minus z_abs_t2"] = (gate_full, z_abs_t1 + s_zflux + hinge_stable + hinge_het + hinge_dagger)
    variants["minus z_abs_t2, minus hinge_dagger"] = (gate_full, z_abs_t1 + s_zflux + hinge_stable + hinge_het)
    variants["minus z_abs_t2, reg-only gate"] = (gate_reg_only, z_abs_t1 + s_zflux + hinge_stable + hinge_het + hinge_dagger)
    variants["minus z_abs_t2, minus hinge_dagger, reg-only gate"] = (gate_reg_only, z_abs_t1 + s_zflux + hinge_stable + hinge_het)
    variants["minus z_abs_t2, plus s(z_dagger) (not hinge)"] = (gate_full, z_abs_t1 + s_zflux + hinge_stable + hinge_het + s(z_dagger))
    variants["minus z_abs_t2, minus hinge_dagger, plus s(z_dagger)"] = (gate_full, z_abs_t1 + s_zflux + hinge_stable + hinge_het + s(z_dagger))
    variants["minus z_abs_t2, half z_abs_t1 weight down (x0.5), plus s_zreg"] = (gate_full, z_abs_t1 + s(z_reg) + s_zflux + hinge_stable + hinge_het + hinge_dagger)

    for name, (gate, raw) in variants.items():
        sc = np.where(gate, raw, -np.inf)
        x, tk, tp, k = full_report(sc, y)
        print(f"  {name:52s} auprc_x={x:.3f}  top-k={tk:.3f} ({tp}/{k})  pass_gate={gate.sum()}")


run("unfiltered", "analytic_infer_yscher_nrand200", "_yscher")
run("annotfilter", "analytic_infer_yscher_annotfilter_nrand200", "_yscher_annotfilter")
