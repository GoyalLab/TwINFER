#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""singletCode singlet counts before vs after applying the RNA-QC threshold
(n_genes >= 1000, pct_counts_mt <= 10) to the lineage-barcode sample sheet,
for endo_T0 / endo_T1. "Before" = all raw barcode-table cellIDs (as in
clone_overlap_test.py). "After" = sample sheet restricted to cellIDs that
pass the RNA-QC threshold in qc_metrics.csv.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
import sys

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
import pandas as pd
from paper_analysis.fatemap_pipeline.fatemap_qc_utils import run_singlet_calling

POOLED_PATH = (
    "/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/sidereaction/stepThreed7/"
    "pooled/stepThreeStarcodeShavedReads_POOLED.txt"
)
QC_CSV = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/qc_metrics.csv'
BARCODE_COL = "BC50StarcodeD7"
MIN_GENES = 1000
MAX_MITO = 10.0


def build_sample_sheet(path, barcode_col=BARCODE_COL):
    df = pd.read_csv(path, sep="\t", usecols=["sample", "cellID", "UMI", barcode_col])
    df = df.drop_duplicates(["sample", "cellID", "UMI", barcode_col]).rename(columns={barcode_col: "barcode"})
    return df[["cellID", "barcode", "sample"]].reset_index(drop=True)


def main():
    sheet = build_sample_sheet(POOLED_PATH)
    qc = pd.read_csv(QC_CSV)
    qc["pass"] = (qc["n_genes"] >= MIN_GENES) & (qc["pct_counts_mt"] <= MAX_MITO)
    pass_keys = set(zip(qc.loc[qc["pass"], "sample"], qc.loc[qc["pass"], "cell_barcode"]))

    print("=== BEFORE RNA-QC threshold (all raw barcode-table cellIDs) ===")
    singlets_before, _, _ = run_singlet_calling(sheet, dataset_name="twinfer_diff_before", min_umi_cutoff=None)
    print(f"\ntotal singlet rows: {len(singlets_before)}")
    print(singlets_before.groupby("sample").size())

    sheet_after = sheet[[(s, c) in pass_keys for c, s in zip(sheet["cellID"], sheet["sample"])]].reset_index(drop=True)
    print(f"\nsample sheet rows before QC filter: {len(sheet)}  after: {len(sheet_after)}")

    print("\n=== AFTER RNA-QC threshold (n_genes>=1000, mito<=10%) ===")
    singlets_after, _, _ = run_singlet_calling(sheet_after, dataset_name="twinfer_diff_after", min_umi_cutoff=None)
    print(f"\ntotal singlet rows: {len(singlets_after)}")
    print(singlets_after.groupby("sample").size())

    print("\n=== SUMMARY ===")
    for sample in ["endo_T0", "endo_T1"]:
        nb = (singlets_before["sample"] == sample).sum()
        na = (singlets_after["sample"] == sample).sum()
        print(f"{sample}: singlets before={nb}  after={na}")


if __name__ == "__main__":
    main()
