#!/usr/bin/env python3
"""Rerun clone_c5 of hPSC_undifferentiated from the stepThree tables with yscher's parameters (HPSC_REPRODUCTION.md section 3; same logic as clones() in yscher's build_endo_dataset.py):
sample sheet = one row per read (cellID, BC50StarcodeD8 -> barcode, sample); singletCode.get_singlets(min_umi_cutoff=5, umi_cutoff_ratio=0, umi_dominant_threshold=30, umi_diff_threshold=0); Singlet cells kept;
barcode per cell = barcode with the most unique UMIs (dedup on cellID, UMI, barcode), ties -> alphabetically first. Compared with clone_c5 of real_data/twinfer2026_run1plus2_counts_mito20_min200genes.h5ad.
usage: hpsc_undiff_clone_c5_rerun.py <T0|T10|T15> [stepThree dir]   (uses yscher's pylibs/singletCode, as the original build)"""
import json
import sys

import anndata as ad
import h5py
import pandas as pd

sys.path.insert(0, "/gpfs/projects/b1255/yscher/Transcriptomic Distance/pylibs")
from singletCode.get_singlets import get_singlets

R = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
tp = sys.argv[1]
S3 = sys.argv[2] if len(sys.argv) > 2 else f"{R}/real_data/hPSC_undifferentiated_barcodes/stepThree"
raw = pd.read_csv(f"{S3}/{tp}/stepThreeStarcodeShavedReads.txt", sep="\t", usecols=["cellID", "UMI", "BC50StarcodeD8"])
raw.columns = ["cellID", "UMI", "barcode"]
print(tp, "reads", len(raw), flush=True)
sh = raw[["cellID", "barcode"]].copy(); sh["sample"] = tp
sl, _ = get_singlets(sh, min_umi_cutoff=5, umi_cutoff_ratio=0.0, umi_dominant_threshold=30, umi_diff_threshold=0)
cells = set(sl.cellID[sl.label == "Singlet"])
u = raw[raw.cellID.isin(cells)].drop_duplicates(["cellID", "UMI", "barcode"]).groupby(["cellID", "barcode"]).size().reset_index(name="n")
c5 = u.sort_values(["cellID", "n", "barcode"], ascending=[True, False, True]).groupby("cellID").barcode.first()
f = h5py.File(f"{R}/real_data/twinfer2026_run1plus2_counts_mito20_min200genes.h5ad", "r"); o = ad.io.read_elem(f["obs"])
o = o[o.timepoint.astype(str) == tp]; o_c5 = o.set_index("cell_id").clone_c5.dropna().astype(str)
mine = c5.reindex(o["cell_id"].astype(str)).dropna()
same = int((mine.values == o_c5.reindex(mine.index).values).sum())
res = dict(timepoint=tp, reads=len(raw), singlet_cells_all=len(cells), our_cells_with_clone_c5=len(o_c5), rerun_cells_with_clone_among_our_cells=len(mine),
           cells_with_clone_equal_sets=bool(set(mine.index) == set(o_c5.index)), only_rerun=len(set(mine.index) - set(o_c5.index)), only_ours=len(set(o_c5.index) - set(mine.index)),
           barcode_strings_identical=same, barcode_strings_compared=len(mine))
print(json.dumps(res), flush=True)
json.dump(res, open(f"{R}/analysis_data/dataset_qc_verification/hpsc_undiff_clone_c5_{tp}.json", "w"), indent=1)
