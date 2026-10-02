"""Test: does existdir's gap to competitors on LARRY close if the magnitude term uses the SAME
cell budget as competitors? Competitors compute plain rho on ALL cells pooled across day2+day4
(~13.5k cells for correlation_high). existdir instead computes rho_t1 (day2 only, clone-weighted)
and rho_t2 (day4 only, clone-weighted) SEPARATELY -- never pooling the two timepoints -- which is
a real sample-size handicap relative to competitors, independent of the twin/clone machinery
(use_clone=True already uses every individual cell, just downweighted by clone size; it does not
restrict to twin-eligible clones).

existdir_pooled = s(|pooled_rho|) + s(gamma), replacing s(|rho_t1|)+s(|rho_t2|) with a single
clone-weighted correlation over the pooled day2+day4 cells -- same cell budget as competitor rho,
same clone-weighting convention as the rest of TwINFER. gamma (direction) is reused unchanged
from the existing twin_score_inputs.csv.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

ROOT = f'{TWINFER_PROJECT_ROOT}'
HERE = f"{ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation"
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f"{HERE}/resources"
R = f"{RES_HERE}/resources"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
from twinfer.inference.correlation_functions import calculate_pairwise_gene_gene_correlation_matrix


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

        raw = pd.read_csv(f"{R}/twinfer_input_cp10k/{gs}.csv")
        pool = raw[raw.time_step.isin([2, 4])].reset_index(drop=True)
        n_pooled = len(pool)
        rho_pooled = calculate_pairwise_gene_gene_correlation_matrix(pool, genes, use_clone=True)

        crit, lvl = gs.rsplit("_", 1)
        panel = gene_sets[gs]
        tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
        dd = d_full[d_full.gene_1.isin(tfs) & (d_full.gene_1 != d_full.gene_2)].reset_index(drop=True)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            continue

        existdir = s(dd.rho_t1.abs().to_numpy()) + s(dd.rho_t2.abs().to_numpy()) + s(dd.gamma.to_numpy())
        pooled_rho_vals = np.array([abs(rho_pooled.loc[a, b]) for a, b in U])
        existdir_pooled = s(pooled_rho_vals) + s(dd.gamma.to_numpy())
        pooled_rho_alone = pooled_rho_vals  # magnitude-only, no direction term at all

        rho_row = br[(br.gene_set == gs) & (br.method == "rho")]
        comp = br[(br.gene_set == gs) & (br.method.isin(["rho", "ppcor", "pidc", "genie3", "grnboost2"]))]
        best_comp = comp.loc[comp.auprc.idxmax()]

        rows.append(dict(
            gene_set=gs, n_pooled_cells=n_pooled, n_pairs=len(U), n_true=int(y.sum()),
            existdir=score_topk(y, existdir),
            existdir_pooled_rho=score_topk(y, existdir_pooled),
            pooled_rho_alone=score_topk(y, pooled_rho_alone),
            competitor_rho=rho_row.auprc.iloc[0] if len(rho_row) else np.nan,
            best_competitor=f"{best_comp.method}({best_comp.auprc:.3f})",
        ))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print(f"\nmean: existdir={df.existdir.mean():.4f}  existdir_pooled_rho={df.existdir_pooled_rho.mean():.4f}  "
          f"pooled_rho_alone={df.pooled_rho_alone.mean():.4f}  competitor_rho={df.competitor_rho.mean():.4f}")
    print(f"existdir_pooled_rho beats existdir on {(df.existdir_pooled_rho>df.existdir).sum()}/{len(df)} gene sets")
    print(f"existdir_pooled_rho beats competitor rho on {(df.existdir_pooled_rho>df.competitor_rho).sum()}/{len(df)} gene sets")
    print(f"pooled_rho_alone vs competitor rho -- should match closely (sanity check): "
          f"corr={np.corrcoef(df.pooled_rho_alone, df.competitor_rho)[0,1]:.3f}")


if __name__ == "__main__":
    main()
