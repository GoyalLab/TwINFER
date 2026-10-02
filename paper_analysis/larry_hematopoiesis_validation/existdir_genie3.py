"""Test: does swapping in GENIE3's mechanism (per-target random forest feature importance,
nonlinear + automatically multivariate) for the magnitude terms close more of the gap than
ppcor's linear partial correlation did? Same architecture as existdir_partial -- computed
separately at t1/t2 (day2-only, day4-only), summed with the unchanged gamma/direction term --
isolating the effect of the deconfounding MECHANISM itself, not sample pooling or clone weighting.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc, precision_recall_curve

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, ".")
from paper_analysis.larry_hematopoiesis_validation.full_analytic_per_geneset import analytic_zfanout

# [2026-10-01 commented out: cwd-relative path; LARRY resources/ now lives in analysis_data (user)]
# R = "./resources"
from twinfer.utils.paths import get_data_root as _res_gdr
R = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation/resources"
RF_KWARGS = {"n_jobs": 1, "n_estimators": 1000, "max_features": "sqrt", "random_state": 0}
NCPU = 8


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def score_topk(y, x):
    prec, rec, _ = precision_recall_curve(y, x)
    return auc(rec, prec)


def _fit_one_target(target, tf_names, X, GI):
    others = [t for t in tf_names if t != target]
    Xtf = X[:, [GI[t] for t in others]]
    y = X[:, GI[target]]
    reg = RandomForestRegressor(**RF_KWARGS)
    reg.fit(Xtf, y)
    imp = dict(zip(others, reg.feature_importances_))
    return target, imp


def genie3_matrix(df_t, genes, tfs):
    gene_cols = [f"{g}_mRNA" for g in genes]
    X = df_t[gene_cols].to_numpy(dtype=float)
    GI = {g: i for i, g in enumerate(genes)}
    out = Parallel(n_jobs=NCPU)(delayed(_fit_one_target)(g, list(tfs), X, GI) for g in genes)
    M = {}
    for target, imp in out:
        for tf, v in imp.items():
            M[(tf, target)] = v
    return M


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    gene_sets = json.load(open(f"{R}/gene_sets.json"))
    detail = json.load(open(f"{R}/gene_sets_detail.json"))
    br = pd.read_csv(f"{R}/benchmark/benchmark_results.csv")

    GENE_SETS = ["variability_high", "variability_mid", "variability_low",
                 "detection_high", "detection_mid", "detection_low",
                 "correlation_high", "correlation_mid", "correlation_low"]
    rows = []
    t0 = time.time()
    for gs in GENE_SETS:
        d_full = pd.read_csv(f"{R}/analytic_infer/{gs}/twin_score_inputs.csv")
        genes = sorted(set(d_full.gene_1) | set(d_full.gene_2))
        zfan_map = analytic_zfanout(d_full, genes)
        raw = pd.read_csv(f"{R}/twinfer_input_cp10k/{gs}.csv")
        t1 = raw[raw.time_step == 2].reset_index(drop=True)
        t2 = raw[raw.time_step == 4].reset_index(drop=True)

        crit, lvl = gs.rsplit("_", 1)
        panel = gene_sets[gs]
        tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
        dd = d_full[d_full.gene_1.isin(tfs) & (d_full.gene_1 != d_full.gene_2)].reset_index(drop=True)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            continue

        g3_t1 = genie3_matrix(t1, genes, tfs)
        g3_t2 = genie3_matrix(t2, genes, tfs)
        g3_1 = np.array([g3_t1.get(p, 0.0) for p in U])
        g3_2 = np.array([g3_t2.get(p, 0.0) for p in U])
        existdir_genie3 = s(g3_1) + s(g3_2) + s(dd.gamma.to_numpy())
        zfan = np.array([zfan_map.get(p, np.nan) for p in U])
        full_genie3 = existdir_genie3 + 1.0 * s(zfan) + 0.5 * s(dd.z_d_het.to_numpy())

        comp = br[(br.gene_set == gs) & (br.method.isin(["rho", "ppcor", "pidc", "genie3", "grnboost2"]))]
        best_comp = comp.loc[comp.auprc.idxmax()]
        genie3_row = br[(br.gene_set == gs) & (br.method == "genie3")]

        rows.append(dict(
            gene_set=gs,
            existdir_genie3=score_topk(y, existdir_genie3),
            full_genie3=score_topk(y, full_genie3),
            competitor_genie3=genie3_row.auprc.iloc[0] if len(genie3_row) else np.nan,
            best_competitor=f"{best_comp.method}({best_comp.auprc:.3f})",
            best_comp_auprc=best_comp.auprc,
        ))
        print(f"[{time.time()-t0:.0f}s] {gs} done", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print(f"\nmean existdir_genie3={df.existdir_genie3.mean():.4f}  full_genie3={df.full_genie3.mean():.4f}  "
          f"competitor_genie3={df.competitor_genie3.mean():.4f}  best_comp={df.best_comp_auprc.mean():.4f}")
    print(f"full_genie3 beats best-per-geneset competitor on {(df.full_genie3>df.best_comp_auprc).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
