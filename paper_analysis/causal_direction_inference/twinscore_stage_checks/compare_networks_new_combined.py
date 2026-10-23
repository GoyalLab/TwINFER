"""New exact combined score |z_rho| - |C/sigma_rho| + |U/sigma_rho| on all network groups (twin-only N = 3000 and 1500), undirected AUPRC x base, vs old combined |z_rho|-|z_C|+|z_U|, TODO4v2 and BEELINE (heatmap values).  Bold = beats TODO4v2 and all four competitors."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, re
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
Z = pd.read_csv(O2 + "twin_decomp_networks.csv"); Z["key"] = [frozenset((int(a), int(b))) for a, b in zip(Z.x, Z.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Z.merge(d[["raw", "sim", "key", "rel"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U", "A_s", "A_u"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int)
Z["new3"] = Z.z_rho.abs() - Z.A_s.abs() + Z.A_u.abs(); Z["old3"] = Z.z_rho.abs() - Z.z_C.abs() + Z.z_U.abs(); Z["rho"] = Z.z_rho.abs()
rows = []
for (N, raw, sim), x in Z.groupby(["N", "raw", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(N=N, group=raw.rsplit("_rep", 1)[0])
    for k in ("new3", "old3", "rho"): r[k] = average_precision_score(x.edge, x[k]) / x.edge.mean()
    rows.append(r)
R = pd.DataFrame(rows); G = R.groupby(["group", "N"])[["new3", "old3", "rho"]].mean().unstack("N"); G.columns = [f"{a}_{b}" for a, b in G.columns]
old = pd.read_csv(O2 + "compare_old_undirected_by_group.csv", index_col=0)[["old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]].dropna(); T = G.join(old, how="inner"); T["best old"] = old.max(axis=1); T["best comp"] = old.drop(columns="old TODO4v2").max(axis=1)
def nm(g):
    m = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)", g)
    if m: n, e, c, a = map(int, m.groups()); return f"n{n} e{e} c{c}" + (" autoreg" if a else "")
    return {"grn_n6_e5_pos100_density": "sweep e5", "grn_n6_e9_pos0_sign_ratio": "sweep e9 all neg", "grn_n6_e9_pos100_center": "sweep e9 all pos", "grn_n6_e9_pos50_sign_ratio": "sweep e9 balanced", "n6_e13_pos100": "sweep e13"}[g]
T["name"] = T.index.map(nm); T = T.sort_index(key=lambda s_: s_.map(lambda q: (2 if "_c" not in q else 0 if q.startswith("grn_n6") else 1, q)))
f = lambda v, b: f"**{v:.2f}**" if b else f"{v:.2f}"
print("| Group | new 3000 | new 1500 | old 3000 | |z_rho| 3000 | TODO4v2 | best competitor |\n|---|---|---|---|---|---|---|")
for g, r in T.iterrows(): print(f"| {r['name']} | {f(r['new3_3000'], r['new3_3000'] > r['best old'])} | {f(r['new3_1500'], r['new3_1500'] > r['best old'])} | {r['old3_3000']:.2f} | {r['rho_3000']:.2f} | {r['old TODO4v2']:.2f} | {r['best comp']:.2f} |")
mx = T.index.str.contains("_c\\d_a"); auto = T.index.str.contains("_a[1-9]\\d*_pos"); cols = ["new3_3000", "new3_1500", "old3_3000", "old3_1500", "rho_3000", "rho_1500", "old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]
M = pd.DataFrame({"mixed no autoreg (8)": T[mx & ~auto][cols].mean(), "mixed autoreg (8)": T[mx & auto][cols].mean(), "sweep (5)": T[~mx][cols].mean(), "all (21)": T[cols].mean()}).round(3); print("\nmean over groups"); print(M.to_string())
print("\ngroups beating ALL old scores:", {k: int((T[k] > T["best old"]).sum()) for k in ("new3_3000", "new3_1500", "old3_3000", "old3_1500", "rho_3000")}, "of", len(T)); print("groups beating TODO4v2:", {k: int((T[k] > T["old TODO4v2"]).sum()) for k in ("new3_3000", "new3_1500", "old3_3000")})
T.round(3).to_csv(O2 + "compare_networks_new_combined_by_group.csv")
