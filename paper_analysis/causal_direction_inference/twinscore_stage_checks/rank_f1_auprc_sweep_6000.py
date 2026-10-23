"""Same ranking check as rank_f1_auprc_n10_6000.py on the NETWORK SWEEP (network_sweep_final sets, 6-gene, 10 simulations x 3 topologies per set), 6,000 twin pairs: |z_U| vs three-term (also two-term, soft, |z_rho|);
AUPRC x base and top-k F1 (k = number of true edge pairs).  TODO4v2 / BEELINE F1 (undirected top-k) from heatmap_data_network_sweep.csv (the 17-edge set has no old values)."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; H = A + "clean_data/benchmarks/network_benchmarks/score_results/"; pd.set_option("display.width", 250)
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "6000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "rel", "fam"]], on=["raw", "sim", "key"]); Z = Z[Z.fam == "sweep"].dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int)
Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs()
SC = {"|z_U|": lambda x: x.azU, "three-term": lambda x: x.azr - x.azC + x.azU, "two-term": lambda x: x.azr - x.azC, "soft": lambda x: x.azr - 0.5 * x.azC + x.azU, "|z_rho|": lambda x: x.azr}
rows = []
for (raw, sim), x in Z.groupby(["raw", "sim"]):
    k = int(x.edge.sum())
    if k in (0, len(x)): continue
    b = x.edge.mean(); r = dict(group=raw.rsplit("_rep", 1)[0], base=b)
    for n, f in SC.items():
        s_ = np.asarray(f(x), float); r[n + " AUPRCxbase"] = average_precision_score(x.edge, s_) / b; r[n + " F1"] = x.edge.to_numpy()[np.argsort(-s_, kind="stable")[:k]].sum() / k
    rows.append(r)
G = pd.DataFrame(rows).groupby("group").mean()
gmap = {"5 edges": "grn_n6_e5_pos100_density", "9 edges": "grn_n6_e9_pos100_center", "13 edges": "n6_e13_pos100", "all negative": "grn_n6_e9_pos0_sign_ratio", "balanced": "grn_n6_e9_pos50_sign_ratio"}
h = pd.read_csv(H + "heatmap_data_network_sweep.csv"); h = h[h.variant == "undirected"].copy(); h["group"] = h.group.replace(gmap)
t4 = h[h.method == "TODO4v2"].groupby("group").f1_topk.mean().rename("TODO4v2 F1"); comp = h[h.method != "TODO4v2"].groupby(["group", "method"]).f1_topk.mean().groupby("group").max().rename("best BEELINE F1"); G = G.join(t4).join(comp)
nm = {"grn_n6_e5_pos100_density": "sweep e5", "grn_n6_e9_pos0_sign_ratio": "sweep e9 all neg", "grn_n6_e9_pos100_center": "sweep e9 all pos", "grn_n6_e9_pos50_sign_ratio": "sweep e9 balanced", "n6_e13_pos100": "sweep e13", "grn_n6_e17_pos100_density": "sweep e17"}
G["name"] = G.index.map(nm); f = lambda v: "–" if pd.isna(v) else f"{v:.2f}"
print("network sweep, 6,000 twin pairs\n| Group | chance (base) | AUPRC×base |z_U| | AUPRC×base three-term | F1 |z_U| | F1 three-term | TODO4v2 F1 | best BEELINE F1 |\n|---|---|---|---|---|---|---|---|")
for g, r in G.iterrows(): print(f"| {r['name']} | {r['base']:.2f} | {f(r['|z_U| AUPRCxbase'])} | {f(r['three-term AUPRCxbase'])} | {f(r['|z_U| F1'])} | {f(r['three-term F1'])} | {f(r['TODO4v2 F1'])} | {f(r['best BEELINE F1'])} |")
cols = [c for c in G.columns if c != "name"]; print("\nmean over the sets:"); print(G[cols].mean().round(3).to_string()); print("\nmean over the 5 sets that have old values:"); print(G.dropna(subset=["TODO4v2 F1"])[cols].mean().round(3).to_string())
print("\nthree-term > |z_U|: AUPRC", int((G["three-term AUPRCxbase"] > G["|z_U| AUPRCxbase"]).sum()), "of", len(G), "| F1", int((G["three-term F1"] > G["|z_U| F1"]).sum()))
G.round(3).to_csv(O2 + "rank_f1_auprc_sweep_6000.csv")
