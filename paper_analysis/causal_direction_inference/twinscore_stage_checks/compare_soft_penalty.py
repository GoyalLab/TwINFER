"""Adds the soft-penalty score |z_rho| - 0.5|z_C| + |z_U| (design X) to the per-group comparison (compare_new_scores_by_group.csv).  Bold = beats TODO4v2 (same sims and old) and all four BEELINE competitors."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, re
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
d = d.dropna(subset=["z_rho", "z_C", "z_U"]).copy(); d["edge"] = d.rel.str.startswith("edge").astype(int); d["azr"] = d.z_rho.abs(); d["azC"] = d.z_C.abs(); d["azU"] = d.z_U.abs()
d["fam"] = d.fam; d = d[~((d.fam == "mixed") & (d.sim > 2))]
rows = []
for (raw, sim), x in d.groupby(["raw", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(group=raw.rsplit("_rep", 1)[0]); base = x.edge.mean()
    for b in (0, 0.5, 1): r[f"b={b}"] = average_precision_score(x.edge, x.azr - b * x.azC + x.azU) / base
    rows.append(r)
S = pd.DataFrame(rows).groupby("group").mean(); C = pd.read_csv(O2 + "compare_new_scores_by_group.csv", index_col=0).join(S, how="inner")
C["best old"] = C[["TODO4v2 (same sims)", "old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]].max(axis=1)
new = ["b=1", "b=0.5", "b=0", "zdag (cross-sample)"]; f = lambda v, b: f"**{v:.2f}**" if b else f"{v:.2f}"
print("| Group | b=1 (|zr|-|zC|+|zU|) | b=0.5 (soft) | b=0 (no C term) | zdag | TODO4v2 | best competitor |\n|---|---|---|---|---|---|---|")
for g, r in C.iterrows(): print(f"| {r['name']} | " + " | ".join(f(r[k], r[k] > r['best old']) for k in new) + f" | {r['TODO4v2 (same sims)']:.2f} | {r[['old PEARSON','old PPCOR','old PIDC','old GRNBOOST2']].max():.2f} |")
mx = C.index.str.contains("_c\\d_a"); auto = C.index.str.contains("_a[1-9]\\d*_pos")
cols = new + ["TODO4v2 (same sims)"]
print("\nmean over groups:"); print(pd.DataFrame({"mixed no autoreg (8)": C[mx & ~auto][cols].mean(), "mixed autoreg (8)": C[mx & auto][cols].mean(), "mixed all (16)": C[mx][cols].mean(), "sweep (5)": C[~mx][cols].mean(), "all (21)": C[cols].mean()}).round(3).to_string())
print("groups beating ALL old scores:", {k: int((C[k] > C["best old"]).sum()) for k in new}, "of", len(C)); print("autoreg groups beating ALL old scores:", {k: int((C[auto][k] > C[auto]["best old"]).sum()) for k in new}, "of", int(auto.sum()))
