"""
Rebuild the TwinScore_supplement summary + D/R/Wz ablation table across all
18 FM06+FM08 gene sets directly from the saved *_pair_terms.csv files.

Bypasses apply_twinscore_supplement_fatemap.py's summary_path
read-modify-write (18 SLURM array tasks writing the same CSV concurrently
dropped rows -- only 14/18 survived after job 7097239). No jobs are rerun;
this only re-derives AUPRC-style stats from already-computed D, R, Wz, PAIR,
TwinScore columns, and sanity-checks every pair_terms.csv for NaN/Inf/very
large numbers first.

Handoff open item 1 (HANDOFF_2026-09-22_fatemap_spacebar_pipeline.md):
"rerun the ablation test ... across all 18 gene sets, not just
FM06/correlation_high".
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

DATA_ROOT = f'{TWINFER_PROJECT_ROOT}/analysis_data'
OUT_DIR = os.path.join(DATA_ROOT, "fatemap_comparison", "data")
BIG_THRESH = 1e4


def full_report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    try:
        auroc = roc_auc_score(y, sc)
    except ValueError:
        auroc = float("nan")
    return dict(auprc=auprc, auprc_random=rand,
                auprc_x=auprc / rand if rand > 0 else float("nan"), auroc=auroc)


def sanity_check(df, fname):
    issues = []
    num = df.select_dtypes(include=[np.number])
    for c in num.columns:
        col = num[c]
        n_nan = int(col.isna().sum())
        n_inf = int(np.isinf(col.to_numpy()).sum())
        finite = col[np.isfinite(col)]
        n_big = int((finite.abs() > BIG_THRESH).sum()) if len(finite) else 0
        if n_nan or n_inf or n_big:
            issues.append(f"{c}: nan={n_nan} inf={n_inf} |x|>{BIG_THRESH:.0e}={n_big}")
    return issues


def main():
    files = sorted(glob.glob(f"{DATA_ROOT}/fm06/data/twinscore_supplement_*_pair_terms.csv") +
                    glob.glob(f"{DATA_ROOT}/fm08/data/twinscore_supplement_*_pair_terms.csv"))
    print(f"found {len(files)} pair_terms files")

    rows = []
    sanity_report = {}
    for f in files:
        base = os.path.basename(f).replace("twinscore_supplement_", "").replace("_pair_terms.csv", "")
        dataset, gs = base.split("_", 1)
        dataset = dataset.upper()

        df = pd.read_csv(f)
        issues = sanity_check(df, base)
        sanity_report[base] = issues

        y = df.collectri_edge.to_numpy()
        score_reports = {
            name: full_report(df[col].to_numpy(), y)
            for name, col in [
                ("TwinScore_full", "TwinScore"),
                ("PAIR_alone", "PAIR"),
                ("D_alone", "D"),
                ("R_alone", "R"),
                ("Wz_alone", "Wz"),
            ]
        }
        row = dict(dataset=dataset, gene_set=gs, n_pairs=len(df), n_true=int(y.sum()),
                   n_sanity_issues=len(issues))
        for name, rep in score_reports.items():
            row[f"{name}_auprc_x"] = rep["auprc_x"]
            row[f"{name}_auroc"] = rep["auroc"]
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(["dataset", "gene_set"]).reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "fatemap_twinscore_ablation_table.csv")
    table.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(table)} rows)")

    print("\n=== sanity check: any NaN / Inf / |x|>1e4 in raw pair_terms columns ===")
    any_issue = False
    for base, issues in sanity_report.items():
        if issues:
            any_issue = True
            print(f"  {base}: " + "; ".join(issues))
    if not any_issue:
        print("  none found in any of the 18 files (all raw columns clean).")

    print("\n=== ablation summary (mean AUPRC-x across all 18 gene sets, by term) ===")
    for name in ["TwinScore_full", "PAIR_alone", "D_alone", "R_alone", "Wz_alone"]:
        col = f"{name}_auprc_x"
        print(f"  {name:16s} mean={table[col].mean():.3f}x  "
              f"median={table[col].median():.3f}x  "
              f"min={table[col].min():.3f}x  max={table[col].max():.3f}x")

    print("\n=== full table ===")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
