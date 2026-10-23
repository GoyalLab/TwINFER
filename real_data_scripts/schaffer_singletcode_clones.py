#!/usr/bin/env python3
"""Schaffer 2026 naive melanoma (GSE279162, Naive1-3): clones from the GEO lineage-barcode counts with the repo's singletCode call
(all samples in ONE call as for Watermelon/FM, singletCode auto UMI cutoff; no cross-sample rescue).
Input : analysis_data/schaffer_lineage_table/Naive{1,2,3}_cell_lineage_long.csv.gz (cell, lineage 'L<n>', umi)  -> expanded to one row per UMI (the singletCode sample-sheet format)
Output: analysis_data/schaffer_lineage_table/singletcode/{singlets.csv, cell_clone.csv, summary.json}"""
import json, os, sys
import numpy as np, pandas as pd
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; D = f"{ROOT}/analysis_data/schaffer_lineage_table"; OUT = f"{D}/singletcode"; os.makedirs(OUT, exist_ok=True)
import singletCode   # singletCode criteria 1-3 incl. its UMI cutoff; the repo's own cross-sample criterion-4 rescue is NOT applied here (user: only the UMI filter)

L = pd.concat([pd.read_csv(f"{D}/Naive{i}_cell_lineage_long.csv.gz") for i in (1, 2, 3)], ignore_index=True)
sheet = L.loc[L.index.repeat(L.umi), ["cell", "lineage", "sample"]].rename(columns={"cell": "cellID", "lineage": "barcode"}).reset_index(drop=True)
print("lineage rows", len(L), "-> sample-sheet rows (one per UMI)", len(sheet), flush=True)
singletCode.check_sample_sheet(sheet)
good, stats = singletCode.get_singlets(sheet, dataset_name="Schaffer_naive")   # min_umi_cutoff not passed: singletCode default floor + auto ratio cutoff
print(stats if not hasattr(stats, "to_string") else stats.to_string(), flush=True)
singlets = good[good["label"] == "Singlet"]; rescued = []
singlet_clone = singlets.groupby(["sample", "cellID"])["barcode"].apply(lambda s: "".join(sorted(set(s))))
singlets.to_csv(f"{OUT}/singlets.csv", index=False)
cc = singlet_clone.rename("clone_barcodes").reset_index()
sizes = cc.groupby("clone_barcodes").size(); cc["clone_size"] = cc.clone_barcodes.map(sizes)
ids = {b: f"Schaffer_clone{i+1:04d}" for i, b in enumerate(sizes.sort_values(ascending=False).index)}; cc.insert(2, "clone", cc.clone_barcodes.map(ids))
cc.to_csv(f"{OUT}/cell_clone.csv", index=False)
nb = L.groupby(["sample", "cell"]).size(); tot = L.groupby("sample").cell.nunique()
summ = {"cells_with_any_barcode": tot.to_dict(), "singlet_cells": cc.groupby("sample").size().to_dict(), "clones_all_samples_pooled": int(cc.clone.nunique()),
        "clones_ge2_cells": int((sizes >= 2).sum()), "largest_clone_cells": int(sizes.max()), "n_cellids_rescued_criterion4": len(set(rescued))}
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=2); print(json.dumps(summ, indent=2))
print(sizes.sort_values(ascending=False).head(10).to_string())
