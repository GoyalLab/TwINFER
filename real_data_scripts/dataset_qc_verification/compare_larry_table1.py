#!/usr/bin/env python3
"""Compare the LARRY final cells under the Table 1 whitelist with the current whitelist (both: corrected rescue, no Weinreb rule), per day."""
import json
import numpy as np
import pandas as pd
R = "/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/rebuild_fullsheet_2026-10-16"
A = pd.read_csv(f"{R}/LARRY_data/filtered/obs_metadata.csv", index_col=0)               # dip test + Otsu
B = pd.read_csv(f"{R}/LARRY_table1_whitelist/filtered/obs_metadata.csv", index_col=0)    # Table 1 minimum UMIs
TB = {"day2": (500, 1720, 1024), "day4": (300, 1993, 1129), "day6": (300, 2211, 1210)}
rows = []
for d, (th, mu, mg) in TB.items():
    n = d[-1]
    a, b = A[A.library.str.contains(f"_d{n}_")], B[B.library.str.contains(f"_d{n}_")]
    rows.append(dict(day=d, Table1_min=th, paper_median_UMI=mu, paper_median_genes=mg,
                     cells_dip_otsu=len(a), cells_Table1=len(b), cells_in_both=len(set(a.index) & set(b.index)),
                     median_UMI_dip_otsu=a.total_counts.median(), median_UMI_Table1=b.total_counts.median(),
                     median_genes_dip_otsu=a.n_genes.median(), median_genes_Table1=b.n_genes.median(),
                     clones_dip_otsu=a.larry_clone_singletcode.replace("", np.nan).nunique(), clones_Table1=b.larry_clone_singletcode.replace("", np.nan).nunique()))
df = pd.DataFrame(rows); pd.set_option("display.width", 250); print(df.T.to_string())
df.to_csv(f"{R}/LARRY_table1_whitelist/compare_vs_dip_otsu.csv", index=False)
print("total final cells: dip+Otsu", len(A), "Table 1", len(B), "in both", len(set(A.index) & set(B.index)))
