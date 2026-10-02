"""Apply the best formula found today (shipped TwinScore's A/B/C/divergence_penalty/
heterogeneity_penalty + GATED*s(z_gamma) [replacing gamma_bonus's sign-only 1/2-weight
reduction] + s(z_reg_gated) [new term]) to all 9 yscher panels matching our variability/
detection/correlation x high/mid/low gene sets, using his exact log1pPF/stable24/newcal/
cloneunit data + his exact ground truth (our own collectri_mouse.tsv, since his copy is not
readable to us -- same source file format, confirmed structurally identical earlier).
Compares against competitors from both his own exports/networks/*.csv (PIDC/GRNBoost2/ppcor/
GENIE3, read directly by his own agreed_full_tables.py loader logic) and, where available,
our own benchmark_results.csv equivalents for context.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

ROOT = "/gpfs/projects/b1255/yscher/Transcriptomic Distance"
YCOPY = f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports"  # [2026-10-01 added: copy of yscher exports (twinscore_script, twinscore_gated, panels/tf_target_panel, _probe_tmp gene_flags)]
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] Cm = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_mouse.tsv', sep="\t")
Cm = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_mouse.tsv', sep="\t")
Cm = Cm[Cm.source_genesymbol != Cm.target_genesymbol]

PANELS = [
    ("perm4clean", "corrhigh", "correlation_high"),
    ("perm4clean", "corrmid", "correlation_mid"),
    ("perm4clean", "corrlow", "correlation_low"),
    ("p4a01", "detect_q90100_tf10", "detection_high"),
    ("p4a01", "detect_q7590_tf10", "detection_mid"),
    ("p4a01", "detect_q5075_tf10", "detection_low"),
    ("p4a01", "spread_q67100_tf10", "variability_high"),
    ("p4a01", "spread_q3367_tf10", "variability_mid"),
    ("p4a01", "spread_q0033_tf10", "variability_low"),
]


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


def load_competitor(path, gset):
    if not os.path.exists(path):
        return None
    cdf = pd.read_csv(path)
    scc = "importance" if "importance" in cdf.columns else cdf.columns[2]
    return {(r[0], r[1]): float(r[2]) for r in cdf[["TF", "target", scc]].itertuples(index=False)
            if r[0] in gset and r[1] in gset}


def main():
    rows = []
    for tag, pn, our_name in PANELS:
        bn = f"larry_log1pPF_stable24_{tag}_{pn}_24"
        # [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] base = f"{ROOT}/exports/twinscore_gated/{bn}"
        base = f"{YCOPY}/twinscore_gated/{bn}"
        d = np.load(f"{base}/frozen_components.npz", allow_pickle=True)
        genes = [str(x) for x in d["genes"]]
        gset = set(genes)
        cur = {(a, b) for a, b in zip(Cm.source_genesymbol, Cm.target_genesymbol)
               if a in gset and b in gset and a != b}
        univ = [(a, b) for a in genes for b in genes if a != b]
        R = sum(1 for p in univ if p in cur)
        if R < 3:
            print(f"skip {pn}: only {R} curated edges")
            continue
        rand = R / len(univ)
        y = np.array([1.0 if p in cur else 0.0 for p in univ])

        T = pd.read_csv(f"{base}/twin_scores.csv")
        Ti = T.set_index([T.gene_1, T.gene_2])

        def col(name):
            return np.array([Ti.loc[p, name] if p in Ti.index else np.nan for p in univ])

        G = pd.read_csv(f"{ROOT}/exports/zreg_gated/{bn}_zreg_gated_t1.csv")
        gated_df = G[G.stat == "gated"]
        zreg_sym = {}
        for r in gated_df.itertuples():
            zreg_sym[(r.gene_1, r.gene_2)] = r.z
            zreg_sym[(r.gene_2, r.gene_1)] = r.z
        z_reg = np.array([zreg_sym.get(p, np.nan) for p in univ])

        A = s(col("z_abs_rho_t1")); B = s(col("z_abs_rho_t2")); Cc = -s(col("z_abs_rho_change"))
        divp = -col("divergence_penalty"); hetp = -col("heterogeneity_penalty")
        s_zg = s(col("z_gamma"))
        gate = (np.abs(s_zg) >= 1.0).astype(float)
        new_gamma = gate * s_zg
        shipped = A + B + Cc + divp + hetp + col("gamma_bonus")
        BEST = A + B + Cc + divp + hetp + new_gamma + s(z_reg)

        auprc_s, auprcx_s, tp_s, nsel_s = report(shipped, y, rand, R)
        auprc_b, auprcx_b, tp_b, nsel_b = report(BEST, y, rand, R)

        comp_files = {
            "PIDC": f"{ROOT}/exports/networks/pidc_{pn}_T4_unrestricted.csv",
            "GRNBoost2": f"{ROOT}/exports/networks/grnboost_{pn}_T4_unrestricted_CellTag-panel.csv",
            "ppcor": f"{ROOT}/exports/networks/ppcor_{pn}_T4_unrestricted.csv",
            "GENIE3": f"{ROOT}/exports/networks/genie3_{pn}_T4_unrestricted.csv",
        }
        best_comp_name, best_comp_auprcx = None, -np.inf
        for nm, fp in comp_files.items():
            m = load_competitor(fp, gset)
            if m is None:
                continue
            sc = np.array([m.get(p, np.nan) for p in univ])
            a_, ax_, t_, n_ = report(s(sc), y, rand, R)
            if ax_ > best_comp_auprcx:
                best_comp_auprcx, best_comp_name = ax_, nm

        rows.append(dict(gene_set=our_name, n_genes=len(genes), n_pairs=len(univ), n_true=R,
                          shipped_auprcx=auprcx_s, shipped_hits=f"{tp_s}/{R}",
                          best_auprcx=auprcx_b, best_hits=f"{tp_b}/{R}",
                          best_competitor=best_comp_name, best_comp_auprcx=best_comp_auprcx))
        print(f"[{our_name}] n_genes={len(genes)} n_true={R} shipped={auprcx_s:.2f}x "
              f"new_best={auprcx_b:.2f}x best_comp={best_comp_name}({best_comp_auprcx:.2f}x)", flush=True)

    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/best_formula_all9_vs_yscher_competitors.csv', index=False)
    df.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/best_formula_all9_vs_yscher_competitors.csv', index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nmean shipped={df.shipped_auprcx.mean():.2f}x  mean new_best={df.best_auprcx.mean():.2f}x  mean best_comp={df.best_comp_auprcx.mean():.2f}x")
    print(f"new_best beats best_comp on {(df.best_auprcx > df.best_comp_auprcx).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
