# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Ablation study on correlation_high (unfiltered twins, yscher panel): which individual raw
features and which formula terms actually carry the CollecTRI signal? Small gene set (44 genes,
1892 pairs) -- safe to run interactively, no SLURM needed.
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
    return auprc / rand if rand > 0 else float("nan"), tp / k if k else float("nan"), tp, k


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

    print(f"\n{'='*70}\n{twin_def}  n_pairs={len(U)} n_true={y.sum()}\n{'='*70}")

    # ---- raw feature signal: each column alone, both signs ----
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy(); z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy(); z_d_het = dd.z_d_het.to_numpy()
    z_gamma = dd.z_gamma.to_numpy(); z_rho_change = dd.z_rho_change.to_numpy()
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    print("\n-- single raw feature, best-of-(+val,-val,|val|) auprc_x --")
    feats = dict(z_abs_rho_t1=z_abs_t1, z_abs_rho_t2=z_abs_t2, z_abs_rho_change=z_abs_change,
                 z_het=z_het, z_div=z_div, z_d_het=z_d_het, z_gamma=z_gamma,
                 z_rho_change=z_rho_change, z_reg_gated=z_reg, z_dagger=z_dagger)
    for name, v in feats.items():
        best_x, best_sign = -np.inf, None
        for sign, label in [(1, "+"), (-1, "-"), (None, "abs")]:
            vv = np.abs(v) if sign is None else sign * v
            x, tk, tp, k = full_report(s(vv), y)
            if x > best_x:
                best_x, best_sign, best_tk, best_tp, best_k = x, label, tk, tp, k
        print(f"  {name:20s} best={best_sign:>3s}  auprc_x={best_x:.3f}  top-k={best_tk:.3f} ({best_tp}/{best_k})")

    # ---- NEW formula: leave-one-term-out ----
    divp = -np.abs(z_div) * (z_het < -2.326).astype(float)
    hetp_old = -z_het * (z_d_het > 0).astype(float)
    hetp_new = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)  # matches new TODO4-style term
    s_zg = s(z_gamma); gate_g = (np.abs(s_zg) >= 1.0).astype(float); new_gamma = gate_g * s_zg
    s_zreg = s(z_reg)
    A, B, Cc = s(z_abs_t1), s(z_abs_t2), -s(z_abs_change)

    NEW_terms = dict(A=A, B=B, Cc=Cc, divp=divp, hetp=hetp_old, new_gamma=new_gamma, s_zreg=s_zreg)
    full_NEW = sum(NEW_terms.values())
    x_full, tk_full, tp_full, k_full = full_report(full_NEW, y)
    print(f"\n-- NEW formula (full): auprc_x={x_full:.3f}  top-k={tk_full:.3f} ({tp_full}/{k_full})")
    print("-- NEW leave-one-out --")
    for name in NEW_terms:
        sc = sum(v for n, v in NEW_terms.items() if n != name)
        x, tk, tp, k = full_report(sc, y)
        print(f"  drop {name:12s} auprc_x={x:.3f} (Δ={x - x_full:+.3f})  top-k={tk:.3f} ({tp}/{k})")

    # ---- TODO4 formula: leave-one-term-out (base terms + hinges), with new hetp ----
    asym_full = dd.assign(asym=dd.rho_cross_xy - dd.rho_cross_yx)
    reg_raw = asym_full.groupby("gene_1")["asym"].mean()
    rv = reg_raw.to_numpy()
    REG = {g: float((reg_raw.get(g, np.nan) - rv.mean()) / max(rv.std(ddof=1), 1e-12)) for g in reg_raw.index}
    z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])
    z_stable = -z_rho_change

    gate = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)
    TODO4_terms = dict(
        z_abs_t1=z_abs_t1, z_abs_t2=z_abs_t2, s_zflux=s(z_flux),
        hinge_stable=-np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0),
        hinge_het=hetp_new,
        hinge_dagger=-np.where(np.abs(z_dagger) > Z_ONE_SIDED, np.abs(z_dagger), 0.0),
    )
    full_TODO4 = np.where(gate, sum(TODO4_terms.values()), -np.inf)
    x_full4, tk_full4, tp_full4, k_full4 = full_report(full_TODO4, y)
    print(f"\n-- TODO4 formula (full, gated): auprc_x={x_full4:.3f}  top-k={tk_full4:.3f} ({tp_full4}/{k_full4})  pass_gate={gate.sum()}")
    print("-- TODO4 leave-one-out (score terms, same gate) --")
    for name in TODO4_terms:
        sc = np.where(gate, sum(v for n, v in TODO4_terms.items() if n != name), -np.inf)
        x, tk, tp, k = full_report(sc, y)
        print(f"  drop {name:14s} auprc_x={x:.3f} (Δ={x - x_full4:+.3f})  top-k={tk:.3f} ({tp}/{k})")

    print("\n-- TODO4 gate ablation --")
    ungated_score = sum(TODO4_terms.values())
    x, tk, tp, k = full_report(ungated_score, y)  # no gate at all
    print(f"  no gate (score on all {len(U)} pairs)       auprc_x={x:.3f}  top-k={tk:.3f} ({tp}/{k})")
    gate_rho_only = np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED
    sc = np.where(gate_rho_only, ungated_score, -np.inf)
    x, tk, tp, k = full_report(sc, y)
    print(f"  |z_rho| gate only (drop z_reg_gated gate)  auprc_x={x:.3f}  top-k={tk:.3f} ({tp}/{k})  pass={gate_rho_only.sum()}")
    gate_reg_only = np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED
    sc = np.where(gate_reg_only, ungated_score, -np.inf)
    x, tk, tp, k = full_report(sc, y)
    print(f"  z_reg_gated gate only (drop |z_rho| gate)  auprc_x={x:.3f}  top-k={tk:.3f} ({tp}/{k})  pass={gate_reg_only.sum()}")


run("unfiltered", "analytic_infer_yscher_nrand200", "_yscher")
run("annotfilter", "analytic_infer_yscher_annotfilter_nrand200", "_yscher_annotfilter")
