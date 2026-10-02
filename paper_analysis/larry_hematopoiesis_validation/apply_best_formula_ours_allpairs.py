"""Apply existdir / shipped TwinScore / NEW formula to OUR 9 gene sets on the FULL
all-gene x all-gene universe (matching yscher's 'unrestricted' convention -- every gene is a
candidate regulator, not just curated TFs), and compare against our OWN allgenes competitor
networks (resources/networks/{method}_{gene_set}_allgenes.csv), a genuine apples-to-apples
universe match this time.
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
Z_HET_THR = -2.326
METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]


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
    boundary = sc[order[k - 1]]
    sel = sc >= boundary
    tp = int((sel & (y == 1)).sum())
    n_sel = int(sel.sum())
    return auprc, auprc / rand, tp, n_sel


def load_competitor_allgenes(method, gs, univ):
    path = f"{R}/networks/{method}_{gs}_allgenes.csv"
    df = pd.read_csv(path)
    sc_col = "importance" if "importance" in df.columns else df.columns[2]
    tf_col, tgt_col = df.columns[0], df.columns[1]
    m = {(r[0], r[1]): float(r[2]) for r in df[[tf_col, tgt_col, sc_col]].itertuples(index=False)}
    return np.array([m.get(p, np.nan) for p in univ])


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    rows = []
    for gs in GENE_SETS:
        d_full = pd.read_csv(f"{R}/analytic_infer/{gs}/twin_score_inputs.csv")
        zdata = json.load(open(f"{R}/infer_results/{gs}_t2_t4_allpairs_50core/z_scores_by_step.json"))
        z_reg_map = zdata["z_reg_gated"]

        def lookup(a, b):
            if f"{a}__{b}" in z_reg_map:
                return z_reg_map[f"{a}__{b}"]
            if f"{b}__{a}" in z_reg_map:
                return z_reg_map[f"{b}__{a}"]
            return np.nan

        dd = d_full  # ALL pairs -- no TF restriction this time
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            continue
        rand = y.mean()
        k = int(y.sum())

        A = s(dd.z_abs_rho_t1.to_numpy()); B = s(dd.z_abs_rho_t2.to_numpy()); Cc = -s(dd.z_abs_rho_change.to_numpy())
        z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy(); z_d_het = dd.z_d_het.to_numpy()
        divp = -np.abs(z_div) * (z_het < Z_HET_THR).astype(float)
        hetp = -z_het * (z_d_het > 0).astype(float)
        s_zg = s(dd.z_gamma.to_numpy())
        gate = (np.abs(s_zg) >= 1.0).astype(float)
        new_gamma = gate * s_zg
        gamma_bonus_orig = 0.5 * np.sign(dd.gamma.to_numpy()) * (np.abs(s_zg) >= 1.0).astype(float)

        z_reg = np.array([lookup(a, b) for a, b in U])
        s_zreg = s(z_reg)

        existdir = s(dd.rho_t1.abs().to_numpy()) + s(dd.rho_t2.abs().to_numpy()) + s(dd.gamma.to_numpy())
        shipped = A + B + Cc + divp + hetp + gamma_bonus_orig
        NEW = A + B + Cc + divp + hetp + new_gamma + s_zreg

        _, ax_ex, _, _ = report(existdir, y, rand, k)
        _, ax_sh, _, _ = report(shipped, y, rand, k)
        auprc_n, ax_n, tp_n, nsel_n = report(NEW, y, rand, k)

        best_comp_name, best_comp_x = None, -np.inf
        for m in METHODS:
            try:
                sc = load_competitor_allgenes(m, gs, U)
            except FileNotFoundError:
                continue
            _, ax_c, _, _ = report(s(sc), y, rand, k)
            if ax_c > best_comp_x:
                best_comp_x, best_comp_name = ax_c, m

        rows.append(dict(gene_set=gs, n_pairs=len(U), n_true=k,
                          existdir_x=ax_ex, shipped_x=ax_sh, new_x=ax_n, new_hits=f"{tp_n}/{k}",
                          best_competitor=best_comp_name, best_comp_x=best_comp_x))
        print(f"[{gs}] n_pairs={len(U)} n_true={k} existdir={ax_ex:.2f}x shipped={ax_sh:.2f}x "
              f"NEW={ax_n:.2f}x ({tp_n}/{k}) best_comp={best_comp_name}({best_comp_x:.2f}x)", flush=True)

    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/best_formula_ours_allpairs_vs_competitors.csv', index=False)
    df.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/best_formula_ours_allpairs_vs_competitors.csv', index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nmean existdir={df.existdir_x.mean():.2f}x shipped={df.shipped_x.mean():.2f}x "
          f"NEW={df.new_x.mean():.2f}x best_comp={df.best_comp_x.mean():.2f}x")
    print(f"NEW beats best_comp on {(df.new_x > df.best_comp_x).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
