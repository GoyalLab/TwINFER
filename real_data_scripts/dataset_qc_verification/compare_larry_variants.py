#!/usr/bin/env python3
"""LARRY: current file (option 1), option 2 (Weinreb rules ii+iii) and option 3 (one copy per duplicated cell barcode + clone): cells, clones, twin pairs, PCA-distance summaries per day (50 PCs of each variant's own 5k-HVG PCA)."""
import numpy as np
import pandas as pd

R = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
V = f"{R}/finalized_data/rebuild_fullsheet_2026-10-16/variants"
SRC = {"opt1_current": f"{R}/analysis_data/dataset_qc_verification/pair_distance/LARRY_pair_distance_summary.csv",
       "opt2_rules_ii_iii": f"{R}/analysis_data/dataset_qc_verification/larry_variants/opt2/pair_distance/LARRY_pair_distance_summary.csv",
       "opt3_one_copy": f"{R}/analysis_data/dataset_qc_verification/larry_variants/opt3/pair_distance/LARRY_pair_distance_summary.csv"}
OBS = {"opt1_current": f"{R}/finalized_data/LARRY_data/filtered/obs_metadata.csv", "opt2_rules_ii_iii": f"{V}/opt2/LARRY_data/filtered/obs_metadata.csv", "opt3_one_copy": f"{V}/opt3/LARRY_data/filtered/obs_metadata.csv"}
rows = []
for k in SRC:
    o = pd.read_csv(OBS[k], index_col=0); o["day"] = "day" + o.library.str.extract(r"_d(\d+)_")[0]
    s = pd.read_csv(SRC[k]); s = s[s.celltype == "ALL"].set_index("sample")
    for d in ("day2", "day4", "day6"):
        c = o[o.day == d].larry_clone_singletcode.astype(str).value_counts()
        rows.append(dict(variant=k, day=d, cells=int((o.day == d).sum()), clones_ge2=int((c >= 2).sum()), twin_pairs=int((c * (c - 1) // 2).sum()),
                         median_dist_clonal=round(s.loc[d, "median_dist_clonal"], 2), median_dist_random=round(s.loc[d, "median_dist_random"], 2),
                         ratio_clonal_over_random=round(s.loc[d, "ratio_clonal_over_random"], 3), AUC_clonal_closer=round(s.loc[d, "auc_clonal_closer"], 3)))
df = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(df.to_string(index=False))
df.to_csv(f"{V}/compare_larry_options.csv", index=False)
# cell-type-level: homotypic pairs, groups present in all three
fr = []
for k in SRC:
    s = pd.read_csv(SRC[k]); s = s[s.celltype != "ALL"]; s["variant"] = k; fr.append(s[["variant", "sample", "celltype", "n_clonal_pairs_available", "median_dist_clonal", "median_dist_random", "ratio_clonal_over_random", "auc_clonal_closer"]])
ct = pd.concat(fr).pivot_table(index=["sample", "celltype"], columns="variant", values=["n_clonal_pairs_available", "ratio_clonal_over_random", "auc_clonal_closer"]).round(3)
print("\ncell-type level (homotypic pairs):"); print(ct.to_string())
ct.to_csv(f"{V}/compare_larry_options_celltypes.csv")
