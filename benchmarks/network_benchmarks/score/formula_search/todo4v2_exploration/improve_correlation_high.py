# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Test concrete candidate variants for improving correlation_high's AUPRC, built directly on
the ablation findings: hetp/z_abs_t2 hurt, z_abs_rho_t1/z_reg_gated/z_dagger carry the signal.
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

    z_abs_t1 = dd.z_abs_rho_t1.to_numpy(); z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy(); z_d_het = dd.z_d_het.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    A, B, Cc = s(z_abs_t1), s(z_abs_t2), -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < -2.326).astype(float)
    hetp_old = -z_het * (z_d_het > 0).astype(float)
    s_zg = s(z_gamma); gate_g = (np.abs(s_zg) >= 1.0).astype(float); new_gamma = gate_g * s_zg
    s_zreg = s(z_reg)
    s_zdagger = s(z_dagger)

    print(f"\n{'='*70}\n{twin_def}  n_pairs={len(U)} n_true={y.sum()}\n{'='*70}")

    variants = {}
    variants["NEW (current, full)"] = A + B + Cc + divp + hetp_old + new_gamma + s_zreg
    variants["NEW minus hetp"] = A + B + Cc + divp + new_gamma + s_zreg
    variants["NEW minus hetp, minus divp"] = A + B + Cc + new_gamma + s_zreg
    variants["NEW minus hetp, half-weight B"] = A + 0.5 * B + Cc + divp + new_gamma + s_zreg
    variants["3-feature: A + s_zreg + s_zdagger"] = A + s_zreg + s_zdagger
    variants["3-feature + B (all 4 strong raw)"] = A + B + s_zreg + s_zdagger
    variants["3-feature, A double-weighted"] = 2 * A + s_zreg + s_zdagger
    variants["NEW minus hetp, plus s_zdagger"] = A + B + Cc + divp + new_gamma + s_zreg + s_zdagger

    for name, sc in variants.items():
        x, tk, tp, k = full_report(sc, y)
        print(f"  {name:38s} auprc_x={x:.3f}  top-k={tk:.3f} ({tp}/{k})")


run("unfiltered", "analytic_infer_yscher_nrand200", "_yscher")
run("annotfilter", "analytic_infer_yscher_annotfilter_nrand200", "_yscher_annotfilter")
