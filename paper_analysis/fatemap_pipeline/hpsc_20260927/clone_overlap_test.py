#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""First test for the twinfer_diff (endo_T0/endo_T1) two-timepoint dataset:
how many clones (singletCode-called) are common to T0 & T1 vs. private to
each timepoint. Uses the POOLED stepThree table (BC50StarcodeD7 collapsed
jointly across both samples) so a shared clone gets an identical barcode
string in both timepoints -- if collapsed per-sample instead, the same true
clone could get slightly different consensus strings in each sample and
this comparison would undercount overlap.
"""
import sys

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
import pandas as pd
from paper_analysis.fatemap_pipeline.fatemap_qc_utils import run_singlet_calling

POOLED_PATH = (
    "/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/sidereaction/stepThreed7/"
    "pooled/stepThreeStarcodeShavedReads_POOLED.txt"
)
BARCODE_COL = "BC50StarcodeD7"


def build_sample_sheet(path, barcode_col=BARCODE_COL):
    df = pd.read_csv(path, sep="\t", usecols=["sample", "cellID", "UMI", barcode_col])
    df = df.drop_duplicates(["sample", "cellID", "UMI", barcode_col]).rename(columns={barcode_col: "barcode"})
    return df[["cellID", "barcode", "sample"]].reset_index(drop=True)


def main():
    sheet = build_sample_sheet(POOLED_PATH)
    print(f"sample sheet rows: {len(sheet)}")
    print(sheet["sample"].value_counts())

    singlets, singlet_clone, rescued_cellids = run_singlet_calling(
        sheet, dataset_name="twinfer_diff", min_umi_cutoff=None
    )
    print(f"\nsinglet rows: {len(singlets)}")
    print(f"distinct cellIDs rescued by criterion-4 cross-sample rescue: {len(set(rescued_cellids))}")

    # singlet_clone index: (sample, cellID) -> clone barcode
    clone_df = singlet_clone.reset_index()
    clone_df.columns = ["sample", "cellID", "clone"]

    clones_by_tp = {tp: set(g["clone"]) for tp, g in clone_df.groupby("sample")}
    t0 = clones_by_tp.get("endo_T0", set())
    t1 = clones_by_tp.get("endo_T1", set())
    shared = t0 & t1

    print(f"\ndistinct clones in endo_T0: {len(t0)}")
    print(f"distinct clones in endo_T1: {len(t1)}")
    print(f"clones common to both (T0 & T1): {len(shared)}")
    print(f"clones private to T0 only: {len(t0 - t1)}")
    print(f"clones private to T1 only: {len(t1 - t0)}")

    n_cells_t0 = clone_df[clone_df["sample"] == "endo_T0"].shape[0]
    n_cells_t1 = clone_df[clone_df["sample"] == "endo_T1"].shape[0]
    n_cells_shared_clones_t0 = clone_df[(clone_df["sample"] == "endo_T0") & (clone_df["clone"].isin(shared))].shape[0]
    n_cells_shared_clones_t1 = clone_df[(clone_df["sample"] == "endo_T1") & (clone_df["clone"].isin(shared))].shape[0]
    print(f"\nsinglet cells in T0: {n_cells_t0} ({n_cells_shared_clones_t0} in shared clones)")
    print(f"singlet cells in T1: {n_cells_t1} ({n_cells_shared_clones_t1} in shared clones)")


if __name__ == "__main__":
    main()
