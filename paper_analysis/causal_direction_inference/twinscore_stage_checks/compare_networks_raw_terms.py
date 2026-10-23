"""Raw (non-normalized) scores on all network groups: |z_U|, |z_rho| - |z_C|, |z_rho| - |z_C| + |z_U|; designs 3,000 twins and 1,500 twins + 3,000 singletons; undirected AUPRC x base (mean over simulations, then over groups).
Old scores as in compare_networks_standardized_3000.py (heatmap undirected AUPRC / GT base).  Bold = beats TODO4v2 and all four competitors.  Output: <O>/compare_networks_raw_terms_by_group.csv, ..._by_network.csv"""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, re
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "3000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z3 = Y.merge(d[["raw", "sim", "key", "rel", "fam"]], on=["raw", "sim", "key"]); ZX = d[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U"]]
SC = {"|z_U|": lambda x: x.azU, "|z_rho|-|z_C|": lambda x: x.azr - x.azC, "|z_rho|-|z_C|+|z_U|": lambda x: x.azr - x.azC + x.azU, "|z_rho| (ref)": lambda x: x.azr}
res = {}
for lab, Z in (("3000 twins", Z3), ("1500+3000 sing", ZX)):
    Z = Z.dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int); Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs(); rows = []
    for (raw, sim), x in Z.groupby(["raw", "sim"]):
        if x.edge.sum() in (0, len(x)): continue
        r = dict(net=raw, group=raw.rsplit("_rep", 1)[0], sim=sim)
        for n, f in SC.items(): r[n] = average_precision_score(x.edge, np.asarray(f(x), float)) / x.edge.mean()
        rows.append(r)
    res[lab] = pd.DataFrame(rows)
G = pd.concat({k: v.groupby("group")[list(SC)].mean() for k, v in res.items()}, axis=1); G.columns = [f"{a}|{b}" for a, b in G.columns]
old = pd.read_csv(O2 + "compare_old_undirected_by_group.csv", index_col=0)[["old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]].dropna()
T = G.join(old, how="inner"); T["best comp"] = old.drop(columns="old TODO4v2").max(axis=1); T["best old"] = old.max(axis=1)
def nm(g):
    m = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)", g)
    if m: n, e, c, a = map(int, m.groups()); return f"n{n} e{e} c{c}" + (" autoreg" if a else "")
    return {"grn_n6_e5_pos100_density": "sweep e5", "grn_n6_e9_pos0_sign_ratio": "sweep e9 all neg", "grn_n6_e9_pos100_center": "sweep e9 all pos", "grn_n6_e9_pos50_sign_ratio": "sweep e9 balanced", "n6_e13_pos100": "sweep e13"}[g]
T["name"] = T.index.map(nm); T = T.sort_index(key=lambda s_: s_.map(lambda q: (2 if "_c" not in q else 0 if q.startswith("grn_n6") else 1, q)))
three = ["|z_U|", "|z_rho|-|z_C|", "|z_rho|-|z_C|+|z_U|"]; f = lambda v, b: f"**{v:.2f}**" if b else f"{v:.2f}"
print("| Group | 3000 tw: U | 3000 tw: rho-C | 3000 tw: rho-C+U | current: U | current: rho-C | current: rho-C+U | TODO4v2 | best competitor |\n|---|---|---|---|---|---|---|---|---|")
for g, r in T.iterrows(): print(f"| {r['name']} | " + " | ".join(f(r[f"{lab}|{k}"], r[f"{lab}|{k}"] > r["best old"]) for lab in ("3000 twins", "1500+3000 sing") for k in three) + f" | {r['old TODO4v2']:.2f} | {r['best comp']:.2f} |")
mx = T.index.str.contains("_c\\d_a"); auto = T.index.str.contains("_a[1-9]\\d*_pos"); cols = [f"{lab}|{k}" for lab in ("3000 twins", "1500+3000 sing") for k in list(SC)]
M = pd.DataFrame({"mixed no autoreg (8)": T[mx & ~auto][cols].mean(), "mixed autoreg (8)": T[mx & auto][cols].mean(), "sweep (5)": T[~mx][cols].mean(), "all (21)": T[cols].mean()}).round(3); print("\nmean over groups:"); print(M.to_string())
print("\nTODO4v2 mean:", T[mx & ~auto]["old TODO4v2"].mean().round(3), T[mx & auto]["old TODO4v2"].mean().round(3), T[~mx]["old TODO4v2"].mean().round(3), T["old TODO4v2"].mean().round(3))
print("groups beating ALL old scores:", {f"{lab}: {k}": int((T[f"{lab}|{k}"] > T["best old"]).sum()) for lab in ("3000 twins", "1500+3000 sing") for k in three}, "of", len(T))
T.round(3).to_csv(O2 + "compare_networks_raw_terms_by_group.csv"); pd.concat({k: v.groupby(["group", "net"])[list(SC)].mean().round(3) for k, v in res.items()}, axis=1).to_csv(O2 + "compare_networks_raw_terms_by_network.csv")
