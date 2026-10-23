"""All network groups (network_sweep_final topology sets + mixed_network_sweep) scored on the 3,000-twin design (20 h sample, no singletons, no cross-sample terms), undirected AUPRC x base, vs TODO4v2 and the BEELINE competitors.
New-metric versions (z-scores from twin_design_zscores.csv, '3000 twins'):
  raw full   : |z_rho| - |z_C| + |z_U|                     raw soft : |z_rho| - 0.5|z_C| + |z_U|
  norm 1:1:1 : s(|z_rho|) - s(|z_C|) + s(|z_U|)             norm 1:.5:.5 : s(|z_rho|) - 0.5 s(|z_C|) + 0.5 s(|z_U|)
  s(v) = (v - mean)/sd(ddof=1) across all gene pairs of the panel (one simulation), as todo4v2.s() does.
Old scores: heatmap_data_{network,mixed}_sweep.csv undirected AUPRC / GT base (= compare_old_auprc_x_undirected.py).  Bold = beats TODO4v2 and all four competitors.
Output: <O>/compare_networks_standardized_3000_by_group.csv, ..._by_network.csv"""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, re
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; H = A + "clean_data/benchmarks/network_benchmarks/score_results/"; B = A + "code/Beeline/inputs/"
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "3000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]
d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "rel", "fam"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U"]); Z["edge"] = Z.rel.str.startswith("edge").astype(int)
Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs()
def s(v): v = np.asarray(v, float); sd = v.std(ddof=1); return (v - v.mean()) / (sd if sd > 0 else 1.0)
V = {"raw full": lambda x: x.azr - x.azC + x.azU, "raw soft": lambda x: x.azr - 0.5 * x.azC + x.azU, "norm 1:1:1": lambda x: s(x.azr) - s(x.azC) + s(x.azU), "norm 1:.5:.5": lambda x: s(x.azr) - 0.5 * s(x.azC) + 0.5 * s(x.azU)}
rows = []
for (raw, sim), x in Z.groupby(["raw", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(net=raw, group=raw.rsplit("_rep", 1)[0], sim=sim)
    for n, f in V.items(): r[n] = average_precision_score(x.edge, np.asarray(f(x), float)) / x.edge.mean()
    rows.append(r)
R = pd.DataFrame(rows); N = R.groupby(["group", "net"])[list(V)].mean().reset_index(); G = R.groupby("group")[list(V)].mean()
old = pd.read_csv(O2 + "compare_old_undirected_by_group.csv", index_col=0)[["old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]].dropna(); T = G.join(old, how="inner")
T["best old"] = T[old.columns].max(axis=1); comp = ["old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]
def nm(g):
    m = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)", g)
    if m: n, e, c, a = map(int, m.groups()); return f"n{n} e{e} c{c}" + (" autoreg" if a else "")
    return {"grn_n6_e5_pos100_density": "sweep e5", "grn_n6_e9_pos0_sign_ratio": "sweep e9 all neg", "grn_n6_e9_pos100_center": "sweep e9 all pos", "grn_n6_e9_pos50_sign_ratio": "sweep e9 balanced", "n6_e13_pos100": "sweep e13"}[g]
T["name"] = T.index.map(nm); T = T.sort_index(key=lambda s_: s_.map(lambda q: (2 if "_c" not in q else 0 if q.startswith("grn_n6") else 1, q)))
f = lambda v, b: f"**{v:.2f}**" if b else f"{v:.2f}"
print("| Group | raw full | raw soft | norm 1:1:1 | norm 1:.5:.5 | TODO4v2 | PEARSON | PPCOR | PIDC | GRNBOOST2 |\n|---|---|---|---|---|---|---|---|---|---|")
for g, r in T.iterrows(): print(f"| {r['name']} | " + " | ".join(f(r[k], r[k] > r['best old']) for k in V) + f" | {r['old TODO4v2']:.2f} | " + " | ".join(f"{r[c]:.2f}" for c in comp) + " |")
mx = T.index.str.contains("_c\\d_a"); auto = T.index.str.contains("_a[1-9]\\d*_pos"); cols = list(V) + list(old.columns)
print("\nmean over groups:"); print(pd.DataFrame({"mixed no autoreg (8)": T[mx & ~auto][cols].mean(), "mixed autoreg (8)": T[mx & auto][cols].mean(), "sweep (5)": T[~mx][cols].mean(), "all (21)": T[cols].mean()}).round(3).to_string())
print("\ngroups beating ALL old scores:", {k: int((T[k] > T['best old']).sum()) for k in V}, "of", len(T)); print("groups beating TODO4v2:", {k: int((T[k] > T['old TODO4v2']).sum()) for k in V})
T.round(3).to_csv(O2 + "compare_networks_standardized_3000_by_group.csv"); N.round(3).to_csv(O2 + "compare_networks_standardized_3000_by_network.csv", index=False)
