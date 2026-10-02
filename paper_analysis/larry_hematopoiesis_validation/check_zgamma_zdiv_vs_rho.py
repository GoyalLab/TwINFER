"""Re-test the Sep-8 best_formula_per_geneset.csv finding (s(z_gamma)-s(z_div) beat rho 2.7x on
detection_high) against rho, existdir, full_gated -- ALL on the SAME twin_score_inputs.csv table
used by today's LARRY existdir/full_gated numbers, so it's apples-to-apples. Answers: what beats
rho on LARRY, and does it hold up across all 9 gene sets (not just the one it was picked on).
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
        crit, lvl = gs.rsplit("_", 1)
        panel = gene_sets[gs]
        tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
        dd = d_full[d_full.gene_1.isin(tfs) & (d_full.gene_1 != d_full.gene_2)].reset_index(drop=True)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            continue

        existdir = s(dd.rho_t1.abs().to_numpy()) + s(dd.rho_t2.abs().to_numpy()) + s(dd.gamma.to_numpy())
        gamma_div = s(dd.z_gamma.to_numpy()) - s(dd.z_div.to_numpy())
        rho_row = br[(br.gene_set == gs) & (br.method == "rho")]
        rho_auprc = rho_row.auprc.iloc[0] if len(rho_row) else np.nan

        rows.append(dict(
            gene_set=gs, n_pairs=len(U), n_true=int(y.sum()),
            rho=rho_auprc,
            existdir=score_topk(y, existdir),
            zgamma_minus_zdiv=score_topk(y, gamma_div),
        ))

    df = pd.DataFrame(rows)
    df["gamma_div_beats_rho"] = df.zgamma_minus_zdiv > df.rho
    df["existdir_beats_rho"] = df.existdir > df.rho
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))
    print(f"\nz_gamma-z_div beats rho on {df.gamma_div_beats_rho.sum()}/{len(df)} gene sets")
    print(f"existdir beats rho on {df.existdir_beats_rho.sum()}/{len(df)} gene sets")
    print(f"mean AUPRC: rho={df.rho.mean():.4f}  existdir={df.existdir.mean():.4f}  zgamma_minus_zdiv={df.zgamma_minus_zdiv.mean():.4f}")


if __name__ == "__main__":
    main()
