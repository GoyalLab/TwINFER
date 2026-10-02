"""TODO.md item 4's twinscore, applied to the 9 gene sets under a given twin definition
(UNFILTERED by default -- pass ANNOTFILTER=1 for the annotation-filtered twin definition once its
z_reg_gated finishes computing).

Two filters (else score = -inf, sunk to the bottom of the ranking):
  max(z_abs_rho_t1, z_abs_rho_t2) > 2.576        ("|z_rho| above 2.576 at either timepoint")
  z_reg_gated > 2.326                             (t1-only; no calibrated t2 version exists yet
                                                    in either infer.py or analytic_zscores.py --
                                                    2026-09-16 decision: use t1 only for now)

Score (for pairs passing both filters):
  z_abs_rho_t1 + z_abs_rho_t2 + s(z_flux)
  z_flux(x, y) = REG(x) - REG(y), REG(g) = panel-z-scored mean over partners w of
                 (rho_cross(g, w) - rho_cross(w, g))   -- "difference in number of targets
                 between the two genes", same REG construction as run_analytic_twinfer_tuned.py

Three hinge terms:
  - I(z_stable > 2.326) * z_stable,           z_stable = z(rho_t1 - rho_t2) = -z_rho_change
  + I(z_het < -2.326) * z_het                 (t1-only, per 2026-09-16 decision)
  - I(|z_dagger| > 2.326) * |z_dagger|,       z_dagger = z-score of rho_cross_xy (the directed
                                               twin cross-time correlation rho^dagger_{x(t1)->y(t2)},
                                               SD_CROSS calibration) -- single quantity, not a
                                               min/max over anything (2026-09-16 decision)

Env:
  ANNOTFILTER=1     use the annotation-filtered twin definition (analytic_infer_annotfilter_nrand200/
                    + infer_results/*_50core_annotfilter/) instead of the unfiltered one.
  P_VALUE=0.01      significance level behind every z-critical threshold below (default 0.01,
                    matching the original 2.326/2.576 pair). |z_rho| and |z_dagger| are two-sided
                    tests (threshold = norm.ppf(1-p/2), e.g. 2.576 at p=0.01); z_reg_gated,
                    z_stable, z_het are one-sided (threshold = norm.ppf(1-p), e.g. 2.326 at
                    p=0.01) -- same one-/two-sided split as the original formula, just
                    parameterized by p instead of hardcoded.
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

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources'
R = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
HERE = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation'
GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]
ANNOTFILTER = os.environ.get("ANNOTFILTER", "0") == "1"
DROP_ZREG_GATE = os.environ.get("DROP_ZREG_GATE", "0") == "1"
P_VALUE = float(os.environ.get("P_VALUE", "0.01"))
# REG_P_VALUE / RHO_P_VALUE screen the z_reg_gated / |z_rho| gate thresholds independently of
# P_VALUE (which still controls |z_dagger|, z_stable, z_het). Each defaults to P_VALUE when
# unset, so passing only P_VALUE keeps every threshold moving together as before.
REG_P_VALUE = float(os.environ.get("REG_P_VALUE", P_VALUE))
RHO_P_VALUE = float(os.environ.get("RHO_P_VALUE", P_VALUE))
ANALYTIC_DIR = "analytic_infer_annotfilter_nrand200" if ANNOTFILTER else "analytic_infer_nrand200"
INFER_SUFFIX = "_50core_annotfilter" if ANNOTFILTER else "_50core"
OUT_CSV = (f"{HERE}/todo4_formula_{'annotfilter' if ANNOTFILTER else 'unfiltered'}"
           f"{'_nozreggate' if DROP_ZREG_GATE else ''}_p{P_VALUE:g}_regp{REG_P_VALUE:g}_rhop{RHO_P_VALUE:g}.csv")

Z_TWO_SIDED = float(norm.ppf(1 - P_VALUE / 2))   # e.g. 2.576 at p=0.01, matches |z_rho|'s original
Z_ONE_SIDED = float(norm.ppf(1 - P_VALUE))       # e.g. 2.326 at p=0.01, matches z_reg_gated's original
Z_REG_ONE_SIDED = float(norm.ppf(1 - REG_P_VALUE))
Z_RHO_TWO_SIDED = float(norm.ppf(1 - RHO_P_VALUE / 2))
Z_RHO_THR = Z_RHO_TWO_SIDED
Z_DAGGER_THR = Z_ONE_SIDED   # original formula used 2.326 (one-sided value) for |z_dagger|, not 2.576
Z_REG_THR = Z_REG_ONE_SIDED
Z_STABLE_THR = Z_ONE_SIDED
Z_HET_THR = -Z_ONE_SIDED


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
    # Direct top-k positional selection (NOT a >=boundary threshold): with a hard gate pushing
    # many scores to -inf, a >=boundary selection collapses to "everything" whenever k exceeds
    # the number of finite (gate-passing) scores, since the k-th largest value is itself -inf --
    # silently inflating "hits" to n_true. Top-k-by-rank avoids that regardless of tie structure.
    order = np.argsort(-sc, kind="stable")
    top_k = order[:k]
    tp = int((y[top_k] == 1).sum())
    return auprc, auprc / rand, tp


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    gene_sets = json.load(open(f"{R}/gene_sets.json"))
    detail = json.load(open(f"{R}/gene_sets_detail.json"))
    bench = pd.read_csv(f"{R}/benchmark/benchmark_results.csv")

    print(f"twin definition: {'ANNOTATION-FILTERED' if ANNOTFILTER else 'UNFILTERED'}  "
          f"({ANALYTIC_DIR}, infer_results/*{INFER_SUFFIX})  "
          f"gate: {'|z_rho| only (z_reg_gated filter REMOVED)' if DROP_ZREG_GATE else '|z_rho| AND z_reg_gated'}  "
          f"p={P_VALUE:g} (|z_dagger|/z_stable/-z_het @ {Z_ONE_SIDED:.3f})  "
          f"reg_p={REG_P_VALUE:g} (z_reg_gated>{Z_REG_THR:.3f})  "
          f"rho_p={RHO_P_VALUE:g} (|z_rho|>{Z_RHO_THR:.3f})\n")

    rows = []
    for gs in GENE_SETS:
        d_full = pd.read_csv(f"{R}/{ANALYTIC_DIR}/{gs}/twin_score_inputs.csv")
        zpath = f"{R}/infer_results/{gs}_t2_t4_allpairs{INFER_SUFFIX}/z_scores_by_step.json"
        if not os.path.exists(zpath):
            print(f"[{gs}] SKIP -- {zpath} not found yet")
            continue
        zdata = json.load(open(zpath))
        z_reg_map = zdata["z_reg_gated"]

        def lookup(a, b):
            if f"{a}__{b}" in z_reg_map:
                return z_reg_map[f"{a}__{b}"]
            if f"{b}__{a}" in z_reg_map:
                return z_reg_map[f"{b}__{a}"]
            return np.nan

        crit, lvl = gs.rsplit("_", 1)
        panel = gene_sets[gs]
        tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
        dd = d_full[d_full.gene_1.isin(tfs) & (d_full.gene_1 != d_full.gene_2)].reset_index(drop=True)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            continue
        rand = y.mean()

        # REG(g): panel-z-scored mean over partners w of (rho_cross(g,w) - rho_cross(w,g)),
        # computed on the FULL pair table (every gene x every gene in this gene set), not just
        # the TF-restricted rows -- matches run_analytic_twinfer_tuned.py's REG construction.
        # asym_full = d_full.assign(asym=d_full.rho_cross_xy - d_full.rho_cross_yx)
        # reg_raw = asym_full.groupby("gene_1")["asym"].mean()
        # rv = reg_raw.to_numpy()
        # REG = {g: float((reg_raw.get(g, np.nan) - rv.mean()) / max(rv.std(ddof=1), 1e-12))
        #        for g in reg_raw.index}
        # 2026-09-17: replaced with yscher's exact "flux_dagger" construction (Transcriptomic
        # Distance/helpers/module_twinscore.py, the "flux_dagger" branch -- the one matching the
        # shipped docstring in helpers/twinscore_model.py): reg(g) = mean_w z_rho_dagger(g->w) -
        # mean_w z_rho_dagger(w->g), built from the NULL-CALIBRATED z-score of rho_cross (not the
        # raw correlation our REG used before), and reg(g) itself is NOT re-standardized across
        # genes -- only the final pair-level z_flux = reg(x) - reg(y) gets standardized (via the
        # existing s(z_flux) below), matching yscher's single standardization pass exactly
        # instead of our old double pass (standardize REG(g), then standardize z_flux again).
        z_dagger_out_full = z_signed(d_full.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])  # g->w
        z_dagger_in_full = z_signed(d_full.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])   # w->g
        asym_full = d_full.assign(z_out=z_dagger_out_full, z_in=z_dagger_in_full)
        regd = asym_full.groupby("gene_1")["z_out"].mean() - asym_full.groupby("gene_1")["z_in"].mean()
        REG = regd.to_dict()

        z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
        z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
        z_reg = np.array([lookup(a, b) for a, b in U])
        z_het = dd.z_het.to_numpy()
        z_stable = -dd.z_rho_change.to_numpy()  # z(rho_t1 - rho_t2) = -z(rho_t2 - rho_t1)
        z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
        z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])

        gate = np.maximum(z_abs_t1, z_abs_t2) > Z_RHO_THR
        if not DROP_ZREG_GATE:
            gate = gate & (np.nan_to_num(z_reg, nan=-np.inf) > Z_REG_THR)

        base = z_abs_t1 + z_abs_t2 + s(z_flux)
        hinge_stable = -np.where(z_stable > Z_STABLE_THR, z_stable, 0.0)
        # hinge_het = np.where(z_het < Z_HET_THR, z_het, 0.0)
        # 2026-09-17: |z_het| > 2.576 (two-sided p=0.01) gate, contributing positively (was a
        # one-sided z_het < -2.326 gate contributing negatively) -- per user instruction.
        hinge_het = np.where(np.abs(z_het) > Z_TWO_SIDED, np.abs(z_het), 0.0)
        hinge_dagger = -np.where(np.abs(z_dagger) > Z_DAGGER_THR, np.abs(z_dagger), 0.0)
        TODO4 = base + hinge_stable + hinge_het + hinge_dagger
        TODO4 = np.where(gate, TODO4, -np.inf)

        auprc, auprcx, tp = report(TODO4, y, rand, int(y.sum()))

        comp = bench[(bench.gene_set == gs) & (bench.method.isin(["rho", "ppcor", "pidc", "genie3", "grnboost2"]))]
        best_comp = comp.loc[comp.auprc.idxmax()]
        best_comp_x = best_comp.auprc / rand

        rows.append(dict(gene_set=gs, n_pairs=len(U), n_true=int(y.sum()), n_pass_gate=int(gate.sum()),
                          todo4_x=auprcx, todo4_hits=f"{tp}/{int(y.sum())}",
                          best_competitor=best_comp.method, best_comp_x=best_comp_x))
        print(f"[{gs}] n_pairs={len(U)} n_true={int(y.sum())} pass_gate={int(gate.sum())} "
              f"TODO4={auprcx:.3f}x ({tp}/{int(y.sum())}) best_comp={best_comp.method}({best_comp_x:.3f}x)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nmean TODO4={df.todo4_x.mean():.3f}x  mean best_comp={df.best_comp_x.mean():.3f}x")
    print(f"TODO4 beats best_comp on {(df.todo4_x > df.best_comp_x).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
