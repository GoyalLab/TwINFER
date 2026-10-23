#!/usr/bin/env python3
"""Clone-size check (is the clone structure dominated by one giant clone?) for LARRY, CellTag3 and Michaels, per timepoint/sample and pooled.
For every group: cells, cells with a clone label, clones, clones with >= 2 cells, the largest clone sizes, share of labelled cells in the largest clone / top 3 / top 10,
within-clone pairs and the share of them in the largest clone, and the second-largest clone with its share of the pairs left after removing the largest.
Outputs: analysis_data/dataset_qc_verification/clone_sizes/clone_size_summary.csv and top_clones.csv"""
import os, numpy as np, pandas as pd, h5py
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; OUT = f"{ROOT}/analysis_data/dataset_qc_verification/clone_sizes"; os.makedirs(OUT, exist_ok=True)
rows = []; tops = []


def stats(ds, group, clone, extra=""):
    clone = pd.Series(clone); n = len(clone); c = clone.dropna().astype(str).value_counts()
    if len(c) == 0: rows.append(dict(dataset=ds, group=group, cells=n, labelled=0)); return
    P = c * (c - 1) / 2; Pt = P.sum(); lab = int(c.sum()); rest = c.iloc[1:]; Pr = (rest * (rest - 1) / 2).sum()
    rows.append(dict(dataset=ds, group=group, note=extra, cells=n, labelled=lab, clones=len(c), clones_ge2=int((c >= 2).sum()), singletons=int((c == 1).sum()), largest=int(c.iloc[0]),
                     second=int(c.iloc[1]) if len(c) > 1 else 0, third=int(c.iloc[2]) if len(c) > 2 else 0, top1_pct_cells=round(100 * c.iloc[0] / lab, 1), top3_pct_cells=round(100 * c.iloc[:3].sum() / lab, 1), top10_pct_cells=round(100 * c.iloc[:10].sum() / lab, 1),
                     within_clone_pairs=int(Pt), top1_pct_pairs=round(100 * P.iloc[0] / Pt, 1) if Pt else np.nan, second_pct_pairs_after_removing_largest=round(100 * P.iloc[1] / Pr, 1) if Pr else np.nan))
    for r, (b, k) in enumerate(c.head(10).items(), 1): tops.append(dict(dataset=ds, group=group, rank=r, cells=int(k), pct_labelled=round(100 * k / lab, 1), clone=str(b)[:40]))


# LARRY: day and sample from the library name (LSK_d2_1, LSK_d4_1_2, ...); clone = singlet clone barcode
L = pd.read_csv(f"{ROOT}/finalized_data/LARRY_data/filtered/obs_metadata.csv", index_col=0); L["day"] = L.library.str.extract(r"_d(\d+)_")[0].astype(int)
L["sample"] = L.library.str.replace(r"^(LSK_d\d+_\d+)(_\d+)?$", r"\1", regex=True)
for d in sorted(L.day.unique()): stats("LARRY", f"day {d}", L[L.day == d].larry_clone_singletcode)
for s in sorted(L["sample"].unique()): stats("LARRY", f"sample {s}", L[L["sample"] == s].larry_clone_singletcode, "libraries pooled")
stats("LARRY", "all days pooled", L.larry_clone_singletcode)
# CellTag3: Time point / experimental_day; published and repaired clone ids; padding cells excluded
f = h5py.File(f"{ROOT}/real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad", "r"); o = f["obs"]
C = pd.DataFrame({"tp": o["Time point"][:], "day": o["experimental_day"][:], "padding": o["is_padding"][:], "pub": o["published_clone"][:], "rep": o["repaired_clone"][:]})
print("CellTag3 cells", len(C), "padding", int(C.padding.sum()), "| time points", sorted(C.tp.unique()), "| experimental_day", sorted(C.day.dropna().unique())[:10])
C = C[~C.padding]
for t in sorted(C.tp.unique()):
    g = C[C.tp == t]; dd = sorted(g.day.dropna().unique())
    for col, nm in (("rep", "repaired clone"), ("pub", "published clone")): stats("CellTag3", f"time point {t} (day {dd[0] if len(dd) == 1 else dd})", g[col], nm)
for col, nm in (("rep", "repaired clone"), ("pub", "published clone")): stats("CellTag3", "all time points pooled", C[col], nm)
# Michaels: two label sets (yscher's cell_to_clone.csv = all labelled cells; the lineage singlets used in the analyses)
M = f"{ROOT}/finalized_data/Michaels_T1T2_data"
A = pd.read_csv(f"{M}/cell_to_clone.csv"); S = pd.read_csv(f"{M}/lineage_singlets_reconstructed_obs.csv", index_col=0)
for t in ("T1", "T2"): stats("Michaels", t, A[A["sample"] == t].clone_id, "cell_to_clone.csv (all labelled cells)"); stats("Michaels", t, S[S.batch == t].clone_id, "lineage singlets (3,948 cells)")
stats("Michaels", "T1+T2 pooled", A.clone_id, "cell_to_clone.csv"); stats("Michaels", "T1+T2 pooled", S.clone_id, "lineage singlets")
R = pd.DataFrame(rows); T = pd.DataFrame(tops); R.to_csv(f"{OUT}/clone_size_summary.csv", index=False); T.to_csv(f"{OUT}/top_clones.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 38)
print(R[["dataset", "group", "note", "cells", "labelled", "clones", "clones_ge2", "largest", "second", "third", "top1_pct_cells", "top10_pct_cells", "within_clone_pairs", "top1_pct_pairs", "second_pct_pairs_after_removing_largest"]].to_string(index=False))
