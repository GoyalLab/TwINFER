"""|z_rho_hat_Delta| (= z_U) on all network groups with 1,500 / 3,000 / 6,000 twin pairs (6,000 = every clone = the full dataset, 12,000 cells; twin-only, nested subsets of one permutation per simulation), undirected AUPRC x base, vs TODO4v2 and BEELINE.
Bold = beats TODO4v2 and all four competitors."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, re
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "rel", "cls"]], on=["raw", "sim", "key"]).dropna(subset=["z_U"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int); Z["azU"] = Z.z_U.abs(); Z["N"] = Z.design.str.extract(r"(\d+)")[0].astype(int)
print("pairs per design:", Z.groupby("N").size().to_dict())
rows = []
for (N, raw, sim), x in Z.groupby(["N", "raw", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    rows.append(dict(N=N, group=raw.rsplit("_rep", 1)[0], u=average_precision_score(x.edge, x.azU) / x.edge.mean()))
R = pd.DataFrame(rows); G = R.groupby(["group", "N"]).u.mean().unstack("N"); G.columns = [f"U_{c}" for c in G.columns]
oldc = pd.read_csv(O2 + "compare_old_undirected_by_group.csv", index_col=0)[["old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]].dropna(); T = G.join(oldc, how="inner"); T["best old"] = oldc.max(axis=1); T["best comp"] = oldc.drop(columns="old TODO4v2").max(axis=1)
def nm(g):
    m = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)", g)
    if m: n, e, c, a = map(int, m.groups()); return f"n{n} e{e} c{c}" + (" autoreg" if a else "")
    return {"grn_n6_e5_pos100_density": "sweep e5", "grn_n6_e9_pos0_sign_ratio": "sweep e9 all neg", "grn_n6_e9_pos100_center": "sweep e9 all pos", "grn_n6_e9_pos50_sign_ratio": "sweep e9 balanced", "n6_e13_pos100": "sweep e13"}[g]
T["name"] = T.index.map(nm); T = T.sort_index(key=lambda s_: s_.map(lambda q: (2 if "_c" not in q else 0 if q.startswith("grn_n6") else 1, q)))
f = lambda v, b: f"**{v:.2f}**" if b else f"{v:.2f}"
print("| Group | 1500 pairs | 3000 pairs | 6000 pairs (12k cells) | TODO4v2 | best competitor |\n|---|---|---|---|---|---|")
for g, r in T.iterrows(): print(f"| {r['name']} | {f(r['U_1500'], r['U_1500'] > r['best old'])} | {f(r['U_3000'], r['U_3000'] > r['best old'])} | {f(r['U_6000'], r['U_6000'] > r['best old'])} | {r['old TODO4v2']:.2f} | {r['best comp']:.2f} |")
mx = T.index.str.contains("_c\\d_a"); auto = T.index.str.contains("_a[1-9]\\d*_pos"); cols = ["U_1500", "U_3000", "U_6000", "old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]
print("\nmean over groups"); print(pd.DataFrame({"mixed no autoreg (8)": T[mx & ~auto][cols].mean(), "mixed autoreg (8)": T[mx & auto][cols].mean(), "sweep (5)": T[~mx][cols].mean(), "all (21)": T[cols].mean()}).round(3).to_string())
print("\ngroups beating ALL old scores:", {k: int((T[k] > T["best old"]).sum()) for k in ("U_1500", "U_3000", "U_6000")}, "of", len(T), "| beating TODO4v2:", {k: int((T[k] > T["old TODO4v2"]).sum()) for k in ("U_1500", "U_3000", "U_6000")})
t = Z.groupby(["N", "cls"]).azU.agg(median="median", pct_gt_196=lambda v: 100 * (v > 1.96).mean()).round(2).unstack("N"); print("\n|z_U| by pair class (median | % > 1.96):"); print(t.reindex(["direct edge", "indirect", "co-regulated", "unrelated"]).to_string())
T.round(3).to_csv(O2 + "compare_networks_zU_fulldata_by_group.csv")
