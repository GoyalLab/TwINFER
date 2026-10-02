"""Reconstruct the per-gene-set numbers behind last night's pooled 0.215 mean-AUPRC claim for
`full` (existdir + s(z_fanout,w=1.0) + 0.5*s(z_d_het), ANALYTIC z_fanout via z_abs_rho_t1,
non-disjoint split) -- that pooled figure was reported without ever printing the per-gene-set
breakdown. Uses the same twin_score_inputs.csv tables (full pairwise, all genes) and same
TF x target restriction as full_gated_fanout_larry.py / check_zgamma_zdiv_vs_rho.py.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f"{HERE}/resources"
R = f"{RES_HERE}/resources"


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


def analytic_zfanout(d_full, genes):
    gx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    Z = np.full((n, n), np.nan)
    for _, row in d_full.iterrows():
        i, j = gx[row.gene_1], gx[row.gene_2]
        Z[i, j] = row.z_abs_rho_t1
    Z = np.where(np.isfinite(Z), Z, np.where(np.isfinite(Z.T), Z.T, np.nan))
    Zsym = np.nanmax(np.stack([Z, Z.T]), axis=0)
    np.fill_diagonal(Zsym, -np.inf)
    zfan = {}
    for _, row in d_full.iterrows():
        a, b = row.gene_1, row.gene_2
        i, j = gx[a], gx[b]
        ci, cj = Zsym[i, :].copy(), Zsym[j, :].copy()
        ci[j] = -np.inf; cj[i] = -np.inf
        with np.errstate(invalid="ignore"):
            cand = np.minimum(ci, cj)
        zfan[(a, b)] = np.nanmax(cand) if np.isfinite(cand).any() else np.nan
    return zfan


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
    for gs in GENE_SETS:
        d_full = pd.read_csv(f"{R}/analytic_infer/{gs}/twin_score_inputs.csv")
        genes = sorted(set(d_full.gene_1) | set(d_full.gene_2))
        zfan_map = analytic_zfanout(d_full, genes)

        crit, lvl = gs.rsplit("_", 1)
        panel = gene_sets[gs]
        tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
        dd = d_full[d_full.gene_1.isin(tfs) & (d_full.gene_1 != d_full.gene_2)].reset_index(drop=True)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            continue

        existdir = s(dd.rho_t1.abs().to_numpy()) + s(dd.rho_t2.abs().to_numpy()) + s(dd.gamma.to_numpy())
        zfan = np.array([zfan_map.get(p, np.nan) for p in U])
        s_zfan = s(zfan)
        s_zdhet = s(dd.z_d_het.to_numpy())
        full = existdir + 1.0 * s_zfan + 0.5 * s_zdhet

        rho_row = br[(br.gene_set == gs) & (br.method == "rho")]
        best_comp_row = br[(br.gene_set == gs) & (br.method.isin(["rho", "ppcor", "pidc", "genie3", "grnboost2"]))]
        best_comp = best_comp_row.loc[best_comp_row.auprc.idxmax()] if len(best_comp_row) else None

        rows.append(dict(
            gene_set=gs, n_pairs=len(U), n_true=int(y.sum()),
            existdir=score_topk(y, existdir),
            full_analytic=score_topk(y, full),
            rho=rho_row.auprc.iloc[0] if len(rho_row) else np.nan,
            best_competitor=f"{best_comp.method}({best_comp.auprc:.3f})" if best_comp is not None else "n/a",
        ))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))
    print(f"\nmean: existdir={df.existdir.mean():.4f}  full_analytic={df.full_analytic.mean():.4f}  rho={df.rho.mean():.4f}")
    print(f"full_analytic beats rho on {(df.full_analytic>df.rho).sum()}/{len(df)} gene sets")


if __name__ == "__main__":
    main()
