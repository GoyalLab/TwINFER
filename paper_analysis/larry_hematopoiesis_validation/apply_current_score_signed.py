"""Signed extension of current_score_allpairs_full_comparison: an edge only counts as a true
positive if BOTH (a) it's a real CollecTRI edge and (b) the predicted regulatory sign matches
CollecTRI's consensus sign. True edges with ambiguous CollecTRI sign (both/neither
stimulation+inhibition) are dropped from the universe entirely (unknown ground truth).

Predicted sign:
  - current_score (twinfer): sign(z_dagger) = sign(rho_cross_xy) -- the cross-time-twin directional
    correlation already computed by the package.
  - all 5 competitors (rho, ppcor, pidc, genie3, grnboost2): all report unsigned
    importance/strength scores (verified: all-positive columns), so per user instruction they are
    each given the sign of the observed correlation, sign(rho_t1), as a common proxy -- the same
    sign value is used for every unsigned method on a given pair.

Same 4 metrics as before (auprc, auprc_x, topk_precision, epr), now computed against the
signed-match label instead of plain existence.
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
    if y.sum() < 1 or y.sum() == len(y):
        return None
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    k = int(y.sum())
    order = np.argsort(-sc, kind="stable")
    top_k = order[:k]
    tp = int((y[top_k] == 1).sum())
    topk_prec = tp / k if k else float("nan")
    return dict(auprc=auprc, auprc_x=auprc / rand if rand > 0 else float("nan"),
                topk_precision=topk_prec, epr=topk_prec / rand if rand > 0 else float("nan"),
                tp=tp, k=k)


def load_zreg_map(gs, infer_suffix):
    path = f"{R}/infer_results/{gs}_t2_t4_allpairs{infer_suffix}/z_scores_by_step.json"
    return json.load(open(path))["z_reg_gated"]


def _reg_and_flux(dd):
    z_dagger_out_full = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_dagger_in_full = z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])
    asym_full = dd.assign(z_out=z_dagger_out_full, z_in=z_dagger_in_full)
    regd = asym_full.groupby("gene_1")["z_out"].mean() - asym_full.groupby("gene_1")["z_in"].mean()
    REG = regd.to_dict()
    return np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])


def current_score(dd):
    z_flux = _reg_and_flux(dd)
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    Cc = -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(z_het < -Z_ONE_SIDED, z_het, 0.0)
    s_zdagger = s(np.abs(z_dagger))

    score = z_abs_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger
    pred_sign = np.sign(dd.rho_cross_xy.to_numpy())
    return score, pred_sign


def load_competitor_allgenes(method, gs, univ):
    path = f"{R}/networks_yscher/{method}_{gs}_allgenes.csv"
    df = pd.read_csv(path)
    m = {(r[0], r[1]): float(r[2]) for r in df[["TF", "target", "importance"]].itertuples(index=False)}
    return np.array([m.get(p, np.nan) for p in univ])


def load_signed_true_edges(ct):
    stim_only = ct.consensus_stimulation & ~ct.consensus_inhibition
    inhib_only = ct.consensus_inhibition & ~ct.consensus_stimulation
    signed = {}
    for row, sgn in [(ct[stim_only], 1), (ct[inhib_only], -1)]:
        for a, b in zip(row.source_genesymbol, row.target_genesymbol):
            signed[(a, b)] = sgn
    return signed


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    SIGNED = load_signed_true_edges(ct)
    print(f"signed true edges (unambiguous): {len(SIGNED)}")

    rows = []
    competitor_cache = {}
    for twin_def, cfg in TWIN_DEFS.items():
        for gs in GENE_SETS:
            dpath = f"{R}/{cfg['analytic_dir']}/{gs}/twin_score_inputs.csv"
            zpath = f"{R}/infer_results/{gs}_t2_t4_allpairs{cfg['infer_suffix']}/z_scores_by_step.json"
            if not os.path.exists(dpath) or not os.path.exists(zpath):
                continue
            dd = pd.read_csv(dpath)
            dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)

            U = list(zip(dd.gene_1, dd.gene_2))
            rho_t1 = dd.rho_t1.to_numpy()
            corr_sign = np.sign(rho_t1)  # "sign of correlation" proxy for unsigned methods

            true_sign = np.array([SIGNED.get(p, 0) for p in U])
            has_truth = true_sign != 0
            if has_truth.sum() < 3:
                continue

            SC, cs_pred_sign = current_score(dd)
            y_signed_cs = (has_truth & (cs_pred_sign == true_sign)).astype(int)
            m_cs = full_report(SC[has_truth], y_signed_cs[has_truth])
            # also score over the FULL universe (unsigned negatives included) -- more realistic
            m_cs_full = full_report(SC, y_signed_cs)
            if m_cs_full is not None:
                rows.append(dict(twin_def=twin_def, gene_set=gs, method="current_score",
                                  n_pairs=len(U), n_signed_true=int(has_truth.sum()),
                                  n_sign_match=int(y_signed_cs.sum()),
                                  auprc=m_cs_full["auprc"], auprc_x=m_cs_full["auprc_x"],
                                  topk_precision=m_cs_full["topk_precision"], epr=m_cs_full["epr"]))

            if gs not in competitor_cache:
                comp = {}
                for meth in METHODS:
                    try:
                        sc = load_competitor_allgenes(meth, gs, U)
                    except FileNotFoundError:
                        continue
                    comp[meth] = sc
                competitor_cache[gs] = comp

            for meth, sc in competitor_cache[gs].items():
                y_signed_m = (has_truth & (corr_sign == true_sign)).astype(int)
                m_m = full_report(s(sc), y_signed_m)
                if m_m is None:
                    continue
                rows.append(dict(twin_def=twin_def, gene_set=gs, method=meth,
                                  n_pairs=len(U), n_signed_true=int(has_truth.sum()),
                                  n_sign_match=int(y_signed_m.sum()),
                                  auprc=m_m["auprc"], auprc_x=m_m["auprc_x"],
                                  topk_precision=m_m["topk_precision"], epr=m_m["epr"]))

            print(f"[{twin_def}/{gs}] n_signed_true={int(has_truth.sum())} "
                  f"n_sign_match(twinfer)={int(y_signed_cs.sum())} "
                  f"current_score_x={m_cs_full['auprc_x'] if m_cs_full else float('nan'):.3f}", flush=True)

    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out_csv = f"{HERE}/current_score_signed_comparison.csv"
    out_csv = f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/current_score_signed_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n" + df.to_string(index=False))

    for metric in ["auprc", "auprc_x", "topk_precision", "epr"]:
        print(f"\n=== SIGNED {metric}: methods (cols) x gene_set (rows), mean over twin_def ===")
        piv = df.pivot_table(index="gene_set", columns="method", values=metric, aggfunc="mean")
        cols = ["current_score"] + [c for c in METHODS if c in piv.columns]
        piv = piv[[c for c in cols if c in piv.columns]]
        print(piv.round(3).to_string())
        print("mean:", piv.mean().round(3).to_dict())

    print("\n=== win counts vs each competitor (SIGNED auprc_x) ===")
    piv_x = df.pivot_table(index=["twin_def", "gene_set"], columns="method", values="auprc_x")
    for meth in METHODS:
        if meth not in piv_x.columns:
            continue
        wins = (piv_x["current_score"] > piv_x[meth]).sum()
        print(f"  current_score beats {meth} on {wins}/{piv_x[meth].notna().sum()}")
    comp_cols = [m for m in METHODS if m in piv_x.columns]
    best_comp = piv_x[comp_cols].max(axis=1)
    wins_best = (piv_x["current_score"] > best_comp).sum()
    print(f"  current_score beats best-of-{len(comp_cols)}-competitors on {wins_best}/{piv_x['current_score'].notna().sum()}")


if __name__ == "__main__":
    main()
