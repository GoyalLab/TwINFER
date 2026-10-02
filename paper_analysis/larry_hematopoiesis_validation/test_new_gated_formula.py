"""Test the new gated formula (ignoring the z_X/fan-out term, per instruction) on all 9 LARRY
gene sets:

  Score(x->y) = max(s(z_abs_rho_t1), s(z_abs_rho_t2)) + s(z_gamma)
                + I(z_het < -2.326) * s(z_het)
                + I(z_reg_gated > 2.326) * s(z_reg_gated)

z_abs_rho_t1/t2, z_gamma, z_het come from the existing twin_score_inputs.csv (already the
correct TF-restricted panel, validated all session). z_reg_gated is NOT in that table -- it's
pulled fresh from resources/infer_results/{gene_set}_t2_t4_allpairs_50core/z_scores_by_step.json,
a genuine production infer_with_twinfer run whose gene keyspace (40 genes per tier) is a
superset of our TF-restricted benchmark panel (verified for detection_high: 12/12 panel genes
found in the 40-gene z_reg_gated keyspace). z_reg_gated/z_het are symmetric (undirected) --
looked up trying both 'a__b' and 'b__a' key orders.

'max_t' in the original formula is only genuinely multi-valued for z_|rho|(t) here (t1 vs t2 are
truly separate quantities in this pipeline); z_het/z_reg_gated are already single combined
per-pair statistics in this implementation (per calculate_twin_score's own docstring: "z_div and
z_het are within-time t1 scores"), so their max_t wrapper is treated as identity.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources'
R = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources'
GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]
Z_HET_THR, Z_REG_THR = -2.326, 2.326


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
    bench = pd.read_csv(f"{R}/benchmark/benchmark_results.csv")

    rows = []
    for gs in GENE_SETS:
        d_full = pd.read_csv(f"{R}/analytic_infer/{gs}/twin_score_inputs.csv")
        zdata = json.load(open(f"{R}/infer_results/{gs}_t2_t4_allpairs_50core/z_scores_by_step.json"))
        z_reg_map, z_het_map = zdata["z_reg_gated"], zdata["z_het"]

        def lookup(m, a, b):
            if f"{a}__{b}" in m:
                return m[f"{a}__{b}"]
            if f"{b}__{a}" in m:
                return m[f"{b}__{a}"]
            return np.nan

        crit, lvl = gs.rsplit("_", 1)
        panel = gene_sets[gs]
        tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
        dd = d_full[d_full.gene_1.isin(tfs) & (d_full.gene_1 != d_full.gene_2)].reset_index(drop=True)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            continue

        s_rho1 = s(dd.z_abs_rho_t1.to_numpy())
        s_rho2 = s(dd.z_abs_rho_t2.to_numpy())
        term_rho = np.maximum(s_rho1, s_rho2)
        term_gamma = s(dd.z_gamma.to_numpy())

        z_het_json = np.array([lookup(z_het_map, a, b) for a, b in U])
        s_zhet = s(z_het_json)
        gate_het = (z_het_json < Z_HET_THR).astype(float)
        term_het = gate_het * s_zhet

        z_reg = np.array([lookup(z_reg_map, a, b) for a, b in U])
        s_zreg = s(z_reg)
        gate_reg = (z_reg > Z_REG_THR).astype(float)
        term_reg = gate_reg * s_zreg

        new_score = term_rho + term_gamma + term_het + term_reg
        existdir = s(dd.rho_t1.abs().to_numpy()) + s(dd.rho_t2.abs().to_numpy()) + s(dd.gamma.to_numpy())

        comp = bench[(bench.gene_set == gs) & (bench.method.isin(["rho", "ppcor", "pidc", "genie3", "grnboost2"]))]
        best_comp = comp.loc[comp.auprc.idxmax()]

        rows.append(dict(
            gene_set=gs, n_pairs=len(U), n_true=int(y.sum()),
            frac_zhet_gated=gate_het.mean(), frac_zreg_gated=gate_reg.mean(),
            auprc_new=score_topk(y, new_score), auprc_existdir=score_topk(y, existdir),
            best_competitor=f"{best_comp.method}({best_comp.auprc:.3f})", best_comp_auprc=best_comp.auprc,
        ))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print(f"\nmean: new={df.auprc_new.mean():.4f}  existdir={df.auprc_existdir.mean():.4f}  best_comp={df.best_comp_auprc.mean():.4f}")
    print(f"new beats existdir on {(df.auprc_new > df.auprc_existdir).sum()}/{len(df)}")
    print(f"new beats best-per-geneset competitor on {(df.auprc_new > df.best_comp_auprc).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
