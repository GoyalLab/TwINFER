"""Full comparison table for yscher's actual gene panels, adding TODO4v2 -- the variant found by
a term-search on correlation_high (2026-09-17): drop z_abs_t2 and the hinge_dagger penalty,
borrow Cc/divp/new_gamma from the NEW formula, add z_dagger as a plain positive signal instead
of a subtracted-penalty hinge. On correlation_high alone this took TODO4 from 1.496x/1.599x
(unfiltered/annotfilter) to 1.741x/1.947x. This script checks whether that generalizes across
all 9 gene sets or is a correlation_high-specific overfit.

TODO4v2 score (same gate as TODO4: max(z_abs_t1,z_abs_t2) > 2.576 AND z_reg_gated > 2.326):
  z_abs_t1 (raw) + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s(z_dagger)
  where Cc, divp, new_gamma are NEW's own terms (see new_formula_score), z_flux/hinge_stable/
  hinge_het are TODO4's own terms unchanged, and s(z_dagger) replaces hinge_dagger.

Also reports NEW and TODO4 (current) and best-of-5-competitor for reference, same as
apply_formulas_yscher_allpairs_with_competitors.py.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources'
R = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
HERE = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]
METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
Z_HET_THR_NEW = -2.326
P_VALUE = 0.01
Z_TWO_SIDED = float(norm.ppf(1 - P_VALUE / 2))
Z_ONE_SIDED = float(norm.ppf(1 - P_VALUE))

TWIN_DEFS = {
    "unfiltered": dict(analytic_dir="analytic_infer_yscher_nrand200", infer_suffix="_yscher"),
    "annotfilter": dict(analytic_dir="analytic_infer_yscher_annotfilter_nrand200", infer_suffix="_yscher_annotfilter"),
}


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
    topk_prec = tp / k if k else float("nan")
    return dict(auprc=auprc, auprc_random=rand, auprc_x=auprc / rand if rand > 0 else float("nan"),
                topk_precision=topk_prec, tp=tp, k=k)


def load_zreg_map(gs, infer_suffix):
    path = f"{R}/infer_results/{gs}_t2_t4_allpairs{infer_suffix}/z_scores_by_step.json"
    return json.load(open(path))["z_reg_gated"]


def lookup(z_reg_map, a, b):
    v = None
    if f"{a}__{b}" in z_reg_map:
        v = z_reg_map[f"{a}__{b}"]
    elif f"{b}__{a}" in z_reg_map:
        v = z_reg_map[f"{b}__{a}"]
    return np.nan if v is None else v


def new_formula_score(dd, U, z_reg_map):
    A = s(dd.z_abs_rho_t1.to_numpy()); B = s(dd.z_abs_rho_t2.to_numpy()); Cc = -s(dd.z_abs_rho_change.to_numpy())
    z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy(); z_d_het = dd.z_d_het.to_numpy()
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    hetp = -z_het * (z_d_het > 0).astype(float)
    s_zg = s(dd.z_gamma.to_numpy())
    gate = (np.abs(s_zg) >= 1.0).astype(float)
    new_gamma = gate * s_zg
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    s_zreg = s(z_reg)
    return A + B + Cc + divp + hetp + new_gamma + s_zreg


def _reg_and_flux(dd):
    """yscher's exact flux_dagger REG construction (used by the reference todo4_score)."""
    z_dagger_out_full = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_dagger_in_full = z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])
    asym_full = dd.assign(z_out=z_dagger_out_full, z_in=z_dagger_in_full)
    regd = asym_full.groupby("gene_1")["z_out"].mean() - asym_full.groupby("gene_1")["z_in"].mean()
    REG = regd.to_dict()
    z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])
    return z_flux


def _reg_and_flux_abs(dd):
    """2026-09-17: TODO4v2-only variant, per user instruction -- REG(g) built from |z_rho_dagger|
    instead of the signed value, so it measures the MAGNITUDE of g's outward vs inward
    directional signal (how strongly g looks like a source vs a target, regardless of the sign
    of the underlying cross-time correlation) rather than yscher's signed net-flow version."""
    z_dagger_out_full = np.abs(z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"]))
    z_dagger_in_full = np.abs(z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"]))
    asym_full = dd.assign(z_out=z_dagger_out_full, z_in=z_dagger_in_full)
    regd = asym_full.groupby("gene_1")["z_out"].mean() - asym_full.groupby("gene_1")["z_in"].mean()
    REG = regd.to_dict()
    z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])
    return z_flux


def todo4_score(dd, U, z_reg_map):
    z_flux = _reg_and_flux(dd)
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    z_het = dd.z_het.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    gate = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)

    base = z_abs_t1 + z_abs_t2 + s(z_flux)
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)
    hinge_dagger = -np.where(np.abs(z_dagger) > Z_ONE_SIDED, np.abs(z_dagger), 0.0)
    score = base + hinge_stable + hinge_het + hinge_dagger
    score = np.where(gate, score, -np.inf)
    return score, int(gate.sum())


def todo4v2_score(dd, U, z_reg_map):
    """OLD variant (reverted 2026-09-17 per Part 2 of HANDOFF_2026-09-17_todo4v2_benchmarks.md):
    signed REG, one-sided z_reg_gated>2.326 gate, one-sided negatively-contributing hinge_het.
    The abs-REG/two-sided-gate/two-sided-hinge_het ("new") variant tested this session regressed
    LARRY's unfiltered+annotfilter means across the board (1.548x/1.540x old vs 1.411x/1.498x new)
    and lost the one row (correlation_high/annotfilter) where old-TODO4v2 beats pidc outright."""
    z_flux = _reg_and_flux(dd)
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    gate = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)

    Cc = -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(z_het < -Z_ONE_SIDED, z_het, 0.0)
    # 2026-09-17: abs-valued per user instruction -- z_dagger's magnitude is a strength-of-
    # regulation/existence signal, its sign is a separate activation-vs-repression call and
    # shouldn't penalize true repressive edges as if they were absent.
    s_zdagger = s(np.abs(z_dagger))

    score = z_abs_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger
    score = np.where(gate, score, -np.inf)
    return score, int(gate.sum())


def load_competitor_allgenes(method, gs, univ):
    path = f"{R}/networks_yscher/{method}_{gs}_allgenes.csv"
    df = pd.read_csv(path)
    m = {(r[0], r[1]): float(r[2]) for r in df[["TF", "target", "importance"]].itertuples(index=False)}
    return np.array([m.get(p, np.nan) for p in univ])


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    rows = []
    competitor_cache = {}
    for twin_def, cfg in TWIN_DEFS.items():
        for gs in GENE_SETS:
            dpath = f"{R}/{cfg['analytic_dir']}/{gs}/twin_score_inputs.csv"
            zpath = f"{R}/infer_results/{gs}_t2_t4_allpairs{cfg['infer_suffix']}/z_scores_by_step.json"
            if not os.path.exists(dpath) or not os.path.exists(zpath):
                print(f"[{twin_def}/{gs}] SKIP -- inputs not found yet")
                continue
            dd = pd.read_csv(dpath)
            dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
            z_reg_map = load_zreg_map(gs, cfg["infer_suffix"])

            U = list(zip(dd.gene_1, dd.gene_2))
            y = np.array([1 if p in CE else 0 for p in U])
            if y.sum() < 3:
                print(f"[{twin_def}/{gs}] SKIP -- only {y.sum()} true edges")
                continue

            NEW = new_formula_score(dd, U, z_reg_map)
            new_m = full_report(NEW, y)

            TODO4, n_pass_gate = todo4_score(dd, U, z_reg_map)
            todo4_m = full_report(TODO4, y)

            TODO4V2, n_pass_gate_v2 = todo4v2_score(dd, U, z_reg_map)
            todo4v2_m = full_report(TODO4V2, y)

            if gs not in competitor_cache:
                best_name, best_m = None, None
                for m in METHODS:
                    try:
                        sc = load_competitor_allgenes(m, gs, U)
                    except FileNotFoundError:
                        continue
                    cm = full_report(s(sc), y)
                    if best_m is None or cm["auprc_x"] > best_m["auprc_x"]:
                        best_name, best_m = m, cm
                competitor_cache[gs] = (best_name, best_m)
            best_name, best_m = competitor_cache[gs]

            rows.append(dict(
                twin_def=twin_def, gene_set=gs, n_pairs=len(U), n_true=int(y.sum()),
                new_auprc_x=new_m["auprc_x"], new_topk_hits=f"{new_m['tp']}/{new_m['k']}",
                todo4_auprc_x=todo4_m["auprc_x"], todo4_topk_hits=f"{todo4_m['tp']}/{todo4_m['k']}",
                todo4v2_auprc=todo4v2_m["auprc"], todo4v2_auprc_random=todo4v2_m["auprc_random"],
                todo4v2_auprc_x=todo4v2_m["auprc_x"], todo4v2_topk_precision=todo4v2_m["topk_precision"],
                todo4v2_topk_hits=f"{todo4v2_m['tp']}/{todo4v2_m['k']}",
                todo4v2_pass_gate=n_pass_gate_v2,
                best_competitor=best_name,
                comp_auprc=best_m["auprc"] if best_m else np.nan,
                comp_auprc_random=best_m["auprc_random"] if best_m else np.nan,
                comp_auprc_x=best_m["auprc_x"] if best_m else np.nan,
                comp_topk_precision=best_m["topk_precision"] if best_m else np.nan,
                comp_topk_hits=f"{best_m['tp']}/{best_m['k']}" if best_m else "",
            ))
            print(f"[{twin_def}/{gs}] n_true={int(y.sum())} "
                  f"NEW={new_m['auprc_x']:.3f}x  TODO4={todo4_m['auprc_x']:.3f}x  "
                  f"TODO4v2={todo4v2_m['auprc_x']:.3f}x ({todo4v2_m['tp']}/{todo4v2_m['k']})  "
                  f"best_comp={best_name}({best_m['auprc_x']:.3f}x)", flush=True)

    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out_csv = f"{HERE}/todo4v2_allpairs_full_comparison.csv"
    out_csv = f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/todo4v2_allpairs_full_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n" + df.to_string(index=False))

    for label, x_col in [("NEW", "new_auprc_x"), ("TODO4", "todo4_auprc_x"), ("TODO4v2", "todo4v2_auprc_x")]:
        piv = df.pivot(index="gene_set", columns="twin_def", values=x_col)
        print(f"\n=== {label} auprc_x, unfiltered vs annotfilter ===")
        print(piv.to_string())
        if "unfiltered" in piv.columns and "annotfilter" in piv.columns:
            print(f"mean: unfiltered={piv['unfiltered'].mean():.3f}x  annotfilter={piv['annotfilter'].mean():.3f}x")

    print("\n=== TODO4 vs TODO4v2 vs best competitor, by twin_def ===")
    for twin_def in TWIN_DEFS:
        sub = df[df.twin_def == twin_def]
        if sub.empty:
            continue
        print(f"[{twin_def}] TODO4 beats best_comp on {(sub.todo4_auprc_x > sub.comp_auprc_x).sum()}/{len(sub)}  "
              f"TODO4v2 beats best_comp on {(sub.todo4v2_auprc_x > sub.comp_auprc_x).sum()}/{len(sub)}  "
              f"TODO4v2 beats TODO4 on {(sub.todo4v2_auprc_x > sub.todo4_auprc_x).sum()}/{len(sub)}  "
              f"mean TODO4={sub.todo4_auprc_x.mean():.3f}x TODO4v2={sub.todo4v2_auprc_x.mean():.3f}x comp={sub.comp_auprc_x.mean():.3f}x")

    print("\n=== per gene set: TODO4 -> TODO4v2 delta ===")
    piv4 = df.pivot(index="gene_set", columns="twin_def", values="todo4_auprc_x")
    piv4v2 = df.pivot(index="gene_set", columns="twin_def", values="todo4v2_auprc_x")
    for gs in GENE_SETS:
        if gs not in piv4.index:
            continue
        u4, u4v2 = piv4.loc[gs, "unfiltered"], piv4v2.loc[gs, "unfiltered"]
        a4, a4v2 = piv4.loc[gs, "annotfilter"], piv4v2.loc[gs, "annotfilter"]
        print(f"  {gs:18s} unfiltered {u4:.3f}->{u4v2:.3f} (Δ{u4v2-u4:+.3f})   "
              f"annotfilter {a4:.3f}->{a4v2:.3f} (Δ{a4v2-a4:+.3f})")


if __name__ == "__main__":
    main()
