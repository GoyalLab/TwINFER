"""Full comparison table for yscher's actual gene panels: the "NEW" and TODO4 TwINFER formulas
under both twin definitions (unfiltered vs annotation-filtered), PLUS the best of 5 competitor
GRN methods (rho, ppcor, pidc, genie3, grnboost2), all on the same unrestricted all-gene x
all-gene universe, against CollecTRI ground truth.

Competitor networks: resources/networks_yscher/{method}_{gene_set}_allgenes.csv -- built this
session directly on yscher's own gene lists (ALL_GENES=1, cp10k-normalized,
twinfer_input_yscher_cp10k), so this is a genuine apples-to-apples comparison, unlike the
earlier-session competitor numbers which were built on our (mismatched) resources/gene_sets.json
panel (see HANDOFF_2026-09-17.md Bug #2). Competitor scores don't depend on the twin definition
(twin pairing only matters for TwINFER's own z-scores), so the same competitor numbers are
reused for both twin_def rows of a given gene set.

For every (twin_def, gene_set) row, reports for NEW and TODO4:
  auprc            raw area under the precision-recall curve
  auprc_random     expected AUPRC of a random ranking (== positive rate, since AUPRC of a
                   random/constant ranking equals the prevalence)
  auprc_x          auprc / auprc_random
  topk_precision   precision within the top-k selection, k = n_true (== early precision)
  topk_hits        "tp/k" string
and the same four for whichever of the 5 competitor methods scores highest by auprc_x on that
gene set (best_competitor names it).
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
    """auprc, auprc_random (== prevalence), auprc_x, topk precision @ k=n_true, tp."""
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


def todo4_score(dd, U, z_reg_map):
    # asym_full = dd.assign(asym=dd.rho_cross_xy - dd.rho_cross_yx)
    # reg_raw = asym_full.groupby("gene_1")["asym"].mean()
    # rv = reg_raw.to_numpy()
    # REG = {g: float((reg_raw.get(g, np.nan) - rv.mean()) / max(rv.std(ddof=1), 1e-12))
    #        for g in reg_raw.index}
    # 2026-09-17: replaced with yscher's exact "flux_dagger" construction (Transcriptomic
    # Distance/helpers/module_twinscore.py "flux_dagger" branch, matching the shipped docstring
    # in helpers/twinscore_model.py): reg(g) = mean_w z_rho_dagger(g->w) - mean_w z_rho_dagger(w->g),
    # built from the null-calibrated z-score of rho_cross (not the raw correlation used before),
    # and reg(g) itself is NOT re-standardized across genes -- only the final z_flux = reg(x) -
    # reg(y) gets standardized (s(z_flux) below), matching yscher's single standardization pass
    # instead of our old double pass (standardize REG(g), then standardize z_flux again).
    z_dagger_out_full = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])  # g->w
    z_dagger_in_full = z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])   # w->g
    asym_full = dd.assign(z_out=z_dagger_out_full, z_in=z_dagger_in_full)
    regd = asym_full.groupby("gene_1")["z_out"].mean() - asym_full.groupby("gene_1")["z_in"].mean()
    REG = regd.to_dict()

    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
    z_het = dd.z_het.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])

    gate = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)

    base = z_abs_t1 + z_abs_t2 + s(z_flux)
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    # hinge_het = np.where(z_het < -Z_ONE_SIDED, z_het, 0.0)
    # 2026-09-17: |z_het| > 2.576 (two-sided p=0.01) gate, contributing positively (was a
    # one-sided z_het < -2.326 gate contributing negatively) -- per user instruction.
    hinge_het = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)
    hinge_dagger = -np.where(np.abs(z_dagger) > Z_ONE_SIDED, np.abs(z_dagger), 0.0)
    score = base + hinge_stable + hinge_het + hinge_dagger
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
    competitor_cache = {}  # gs -> (best_name, metrics dict), independent of twin_def
    for twin_def, cfg in TWIN_DEFS.items():
        for gs in GENE_SETS:
            dpath = f"{R}/{cfg['analytic_dir']}/{gs}/twin_score_inputs.csv"
            zpath = f"{R}/infer_results/{gs}_t2_t4_allpairs{cfg['infer_suffix']}/z_scores_by_step.json"
            if not os.path.exists(dpath) or not os.path.exists(zpath):
                print(f"[{twin_def}/{gs}] SKIP -- inputs not found yet ({dpath if not os.path.exists(dpath) else zpath})")
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
                new_auprc=new_m["auprc"], new_auprc_random=new_m["auprc_random"], new_auprc_x=new_m["auprc_x"],
                new_topk_precision=new_m["topk_precision"], new_topk_hits=f"{new_m['tp']}/{new_m['k']}",
                todo4_auprc=todo4_m["auprc"], todo4_auprc_random=todo4_m["auprc_random"], todo4_auprc_x=todo4_m["auprc_x"],
                todo4_topk_precision=todo4_m["topk_precision"], todo4_topk_hits=f"{todo4_m['tp']}/{todo4_m['k']}",
                todo4_pass_gate=n_pass_gate,
                best_competitor=best_name,
                comp_auprc=best_m["auprc"] if best_m else np.nan,
                comp_auprc_x=best_m["auprc_x"] if best_m else np.nan,
                comp_topk_precision=best_m["topk_precision"] if best_m else np.nan,
                comp_topk_hits=f"{best_m['tp']}/{best_m['k']}" if best_m else "",
            ))
            print(f"[{twin_def}/{gs}] n_pairs={len(U)} n_true={int(y.sum())} "
                  f"NEW={new_m['auprc_x']:.3f}x (top-k prec {new_m['topk_precision']:.3f}, {new_m['tp']}/{new_m['k']})  "
                  f"TODO4={todo4_m['auprc_x']:.3f}x (top-k prec {todo4_m['topk_precision']:.3f}, {todo4_m['tp']}/{todo4_m['k']})  "
                  f"best_comp={best_name}({best_m['auprc_x']:.3f}x, top-k prec {best_m['topk_precision']:.3f})",
                  flush=True)

    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out_csv = f"{HERE}/formulas_yscher_allpairs_full_comparison.csv"
    out_csv = f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/formulas_yscher_allpairs_full_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n" + df.to_string(index=False))

    for label, x_col, tk_col in [("NEW", "new_auprc_x", "new_topk_precision"),
                                  ("TODO4", "todo4_auprc_x", "todo4_topk_precision")]:
        piv = df.pivot(index="gene_set", columns="twin_def", values=x_col)
        print(f"\n=== {label} auprc_x, unfiltered vs annotfilter ===")
        print(piv.to_string())
        if "unfiltered" in piv.columns and "annotfilter" in piv.columns:
            print(f"mean: unfiltered={piv['unfiltered'].mean():.3f}x  annotfilter={piv['annotfilter'].mean():.3f}x  "
                  f"annotfilter wins on {(piv['annotfilter'] > piv['unfiltered']).sum()}/{piv.dropna().shape[0]}")

    print("\n=== best competitor per gene set (twin-def independent) ===")
    comp_summary = df.drop_duplicates("gene_set")[["gene_set", "best_competitor", "comp_auprc_x", "comp_topk_precision"]]
    print(comp_summary.to_string(index=False))

    print("\n=== NEW / TODO4 vs best competitor, by twin_def ===")
    for twin_def in TWIN_DEFS:
        sub = df[df.twin_def == twin_def]
        if sub.empty:
            continue
        print(f"[{twin_def}] NEW beats best_comp on {(sub.new_auprc_x > sub.comp_auprc_x).sum()}/{len(sub)}  "
              f"TODO4 beats best_comp on {(sub.todo4_auprc_x > sub.comp_auprc_x).sum()}/{len(sub)}  "
              f"mean NEW={sub.new_auprc_x.mean():.3f}x TODO4={sub.todo4_auprc_x.mean():.3f}x comp={sub.comp_auprc_x.mean():.3f}x")


if __name__ == "__main__":
    main()
