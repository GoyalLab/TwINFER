#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""AUPRC x baseline (per memory: always AUPRC x base rate, never AUROC) for HS054_Sot48h:
TwinScore_supplement (gated bootstrap), analytic twinScore (default calculate_twin_score,
already in ranked_edges), and 5 competitors (rho/ppcor/pidc/genie3/grnboost2), vs human
CollecTRI, per gene set -- whatever has finished so far."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

COLLECTRI_PATH = (
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
    f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
)
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data'
SUPP_DIR = f"{DATA_DIR}/twinscore_supp_gated_bootstrap"
NET_DIR = f"{DATA_DIR}/networks"
GENE_SETS = [
    "variability_high", "variability_mid", "variability_low",
    "detection_high", "detection_mid", "detection_low",
    "correlation_high", "correlation_mid", "correlation_low",
]
METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]


def full_report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    k = int(y.sum())
    order = np.argsort(-sc, kind="stable")
    tp = int((y[order[:k]] == 1).sum())
    topk_prec = tp / k if k else float("nan")
    return dict(auprc=auprc, auprc_random=rand, auprc_x=auprc / rand if rand > 0 else float("nan"),
                topk_precision=topk_prec, tp=tp, k=k)


def load_competitor(method, gs, univ):
    path = f"{NET_DIR}/{method}_{gs}_allgenes.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    m = {(r[0], r[1]): float(r[2]) for r in df[["TF", "target", "importance"]].itertuples(index=False)}
    return np.array([m.get(p, np.nan) for p in univ])


def main():
    ct = pd.read_csv(COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    print(f"human CollecTRI: {len(CE):,} directed edges\n", flush=True)

    rows = []
    for gs in GENE_SETS:
        re_path = f"{DATA_DIR}/ranked_edges_hs054_{gs}.csv"
        supp_path = f"{SUPP_DIR}/twinscore_supplement_hs054_{gs}_gated_bootstrap_pair_terms.csv"
        if not (os.path.exists(re_path) and os.path.exists(supp_path)):
            print(f"[{gs}] SKIP -- not finished yet", flush=True)
            continue

        dd = pd.read_csv(re_path)
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            print(f"[{gs}] SKIP -- only {int(y.sum())} true CollecTRI edges in {len(U)} candidates", flush=True)
            continue

        row = dict(gene_set=gs, n_pairs=len(U), n_true=int(y.sum()))

        m_analytic = full_report(dd["twinScore"].to_numpy(), y)
        row["twinScore_analytic_auprc_x"] = m_analytic["auprc_x"]
        row["twinScore_analytic_topk_prec"] = m_analytic["topk_precision"]

        supp = pd.read_csv(supp_path)
        supp_map = {(a, b): v for a, b, v in zip(supp.gene_1, supp.gene_2, supp.TwinScore)}
        sc_supp = np.array([supp_map.get(p, np.nan) for p in U])
        m_supp = full_report(sc_supp, y)
        row["twinscore_supp_auprc_x"] = m_supp["auprc_x"]
        row["twinscore_supp_topk_prec"] = m_supp["topk_precision"]

        for method in METHODS:
            sc = load_competitor(method, gs, U)
            if sc is None:
                row[f"{method}_auprc_x"] = np.nan
                row[f"{method}_topk_prec"] = np.nan
                continue
            m = full_report(sc, y)
            row[f"{method}_auprc_x"] = m["auprc_x"]
            row[f"{method}_topk_prec"] = m["topk_precision"]

        rows.append(row)
        print(f"[{gs}] n_true={int(y.sum())}/{len(U)}  "
              f"twinScore(analytic)={row['twinScore_analytic_auprc_x']:.2f}x  "
              f"twinscore_supp={row['twinscore_supp_auprc_x']:.2f}x  "
              + "  ".join(f"{m}={row[f'{m}_auprc_x']:.2f}x" for m in METHODS), flush=True)

    df = pd.DataFrame(rows)
    out_csv = f"{DATA_DIR}/hs054_vs_collectri_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}\n", flush=True)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False), flush=True)

    print("\n=== mean AUPRC x baseline across completed gene sets ===", flush=True)
    for col in ["twinScore_analytic", "twinscore_supp"] + METHODS:
        c = f"{col}_auprc_x"
        if c in df.columns:
            print(f"  {col:>20s}: {df[c].mean():.3f}x  (n={df[c].notna().sum()} gene sets)", flush=True)


if __name__ == "__main__":
    main()
