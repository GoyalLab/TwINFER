"""Budget-matched design: 6,000 twin pairs (all 12,000 cells at 20 h) vs TODO4v2 / BEELINE (12,000 cells split over 1 h and 20 h in the TwINFER partition) and cross-sample coupling.  Scores at 6,000 pairs: |z_U|, |z_rho|-|z_C|, |z_rho|-|z_C|+|z_U| (old terms, raw full), soft (b=.5), and normalized 1:.5:.5 (panel z-scored)."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, re
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "6000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "rel"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int); Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs()
def s(v): v = np.asarray(v, float); sd = v.std(ddof=1); return (v - v.mean()) / (sd if sd > 0 else 1.0)
V = {"|z_U|": lambda x: x.azU, "|z_rho|": lambda x: x.azr, "rho-C": lambda x: x.azr - x.azC, "rho-C+U (full)": lambda x: x.azr - x.azC + x.azU, "soft b=.5": lambda x: x.azr - 0.5 * x.azC + x.azU, "norm 1:.5:.5": lambda x: s(x.azr) - 0.5 * s(x.azC) + 0.5 * s(x.azU)}
rows = []
for (raw, sim), x in Z.groupby(["raw", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(group=raw.rsplit("_rep", 1)[0])
    for k, f in V.items(): r[k] = average_precision_score(x.edge, np.asarray(f(x), float)) / x.edge.mean()
    rows.append(r)
G = pd.DataFrame(rows).groupby("group").mean(); oldc = pd.read_csv(O2 + "compare_old_undirected_by_group.csv", index_col=0)[["old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]].dropna()
zc = pd.read_csv(O2 + "compare_new_scores_by_group.csv", index_col=0)[["zdag (cross-sample)"]].rename(columns={"zdag (cross-sample)": "cross-sample"}); T = G.join(oldc, how="inner").join(zc); T["best old"] = oldc.max(axis=1)
mx = T.index.str.contains("_c\\d_a"); auto = T.index.str.contains("_a[1-9]\\d*_pos"); cols = list(V) + ["cross-sample", "old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]
print(pd.DataFrame({"mixed no autoreg (8)": T[mx & ~auto][cols].mean(), "mixed autoreg (8)": T[mx & auto][cols].mean(), "sweep (5)": T[~mx][cols].mean(), "all (21)": T[cols].mean()}).round(3).to_string())
print("\ngroups beating ALL old scores (of 21):", {k: int((T[k] > T["best old"]).sum()) for k in list(V) + ["cross-sample"]}); print("groups beating TODO4v2:", {k: int((T[k] > T["old TODO4v2"]).sum()) for k in list(V) + ["cross-sample"]})
T.round(3).to_csv(O2 + "compare_networks_6000_combined_by_group.csv")
