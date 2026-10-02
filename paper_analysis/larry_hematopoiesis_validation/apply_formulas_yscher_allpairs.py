"""Apply both the "NEW" formula (apply_best_formula_ours_allpairs.py) and the TODO4 formula
(apply_todo4_formula.py) to yscher's ACTUAL gene panels (resources/gene_sets_yscher.json genes,
already baked into resources/twinfer_input_yscher_cp10k*), on the FULL all-gene x all-gene
universe (yscher's own convention -- no TF/target split), under BOTH twin definitions
(unfiltered vs annotation-filtered), against CollecTRI ground truth.

This answers TODO 2.1 (does annotation-filtering twins help?) on a panel that is no longer a
confound (see HANDOFF_2026-09-17.md Bug #2): every gene set here is yscher's own gene list, not
our (previously mismatched) resources/gene_sets.json panel of the same name.

No competitor (rho/ppcor/pidc/genie3/grnboost2) column is reported here: those networks in
resources/networks/ were built on OUR gene panels, not yscher's, so comparing them against
yscher-panel scores would reintroduce the exact panel-mismatch bug this run exists to avoid.
This script is TwINFER-vs-itself only (unfiltered vs annotation-filtered).

Env:
  N_RAND  must match whichever analytic_infer_yscher*_nrand200 dirs exist (default 200, matches
          the dirs built this session).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources'
R = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
HERE = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation'
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]
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


def report(sc, y, rand, k):
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    order = np.argsort(-sc, kind="stable")
    top_k = order[:k]
    tp = int((y[top_k] == 1).sum())
    return auprc, auprc / rand, tp


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


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    rows = []
    for twin_def, cfg in TWIN_DEFS.items():
        for gs in GENE_SETS:
            dpath = f"{R}/{cfg['analytic_dir']}/{gs}/twin_score_inputs.csv"
            if not os.path.exists(dpath):
                print(f"[{twin_def}/{gs}] SKIP -- {dpath} not found")
                continue
            zpath = f"{R}/infer_results/{gs}_t2_t4_allpairs{cfg['infer_suffix']}/z_scores_by_step.json"
            if not os.path.exists(zpath):
                print(f"[{twin_def}/{gs}] SKIP -- {zpath} not found yet")
                continue
            dd = pd.read_csv(dpath)
            dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
            z_reg_map = load_zreg_map(gs, cfg["infer_suffix"])

            U = list(zip(dd.gene_1, dd.gene_2))
            y = np.array([1 if p in CE else 0 for p in U])
            if y.sum() < 3:
                print(f"[{twin_def}/{gs}] SKIP -- only {y.sum()} true edges")
                continue
            rand = y.mean()
            k = int(y.sum())

            NEW = new_formula_score(dd, U, z_reg_map)
            _, new_x, new_tp = report(NEW, y, rand, k)

            TODO4, n_pass_gate = todo4_score(dd, U, z_reg_map)
            _, todo4_x, todo4_tp = report(TODO4, y, rand, k)

            rows.append(dict(twin_def=twin_def, gene_set=gs, n_pairs=len(U), n_true=k,
                              new_x=new_x, new_hits=f"{new_tp}/{k}",
                              todo4_x=todo4_x, todo4_hits=f"{todo4_tp}/{k}", todo4_pass_gate=n_pass_gate))
            print(f"[{twin_def}/{gs}] n_pairs={len(U)} n_true={k} "
                  f"NEW={new_x:.3f}x ({new_tp}/{k})  TODO4={todo4_x:.3f}x ({todo4_tp}/{k}, pass_gate={n_pass_gate})",
                  flush=True)

    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f"{HERE}/formulas_yscher_allpairs_unfiltered_vs_annotfilter.csv", index=False)
    df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/formulas_yscher_allpairs_unfiltered_vs_annotfilter.csv", index=False)
    print("\n" + df.to_string(index=False))

    piv_new = df.pivot(index="gene_set", columns="twin_def", values="new_x")
    piv_todo4 = df.pivot(index="gene_set", columns="twin_def", values="todo4_x")
    print("\n=== NEW formula, unfiltered vs annotfilter (x random) ===")
    print(piv_new.to_string())
    print(f"mean: unfiltered={piv_new['unfiltered'].mean():.3f}x  annotfilter={piv_new['annotfilter'].mean():.3f}x  "
          f"annotfilter wins on {(piv_new['annotfilter'] > piv_new['unfiltered']).sum()}/{len(piv_new)}")

    print("\n=== TODO4 formula, unfiltered vs annotfilter (x random) ===")
    print(piv_todo4.to_string())
    print(f"mean: unfiltered={piv_todo4['unfiltered'].mean():.3f}x  annotfilter={piv_todo4['annotfilter'].mean():.3f}x  "
          f"annotfilter wins on {(piv_todo4['annotfilter'] > piv_todo4['unfiltered']).sum()}/{len(piv_todo4)}")


if __name__ == "__main__":
    main()
