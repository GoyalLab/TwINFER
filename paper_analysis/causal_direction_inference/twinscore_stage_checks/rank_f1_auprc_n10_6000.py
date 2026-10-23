"""Ranking check, 6,000 twin pairs, 10-gene mixed networks (and 6-gene for reference): |z_U| (= |z_rho_hat_Delta|) vs the three-term score |z_rho|-|z_C|+|z_U| (also two-term, soft, |z_rho|).
Metrics per (network, sim): AUPRC x base (undirected edge pairs) and top-k F1 (k = number of true edge pairs; with k predictions precision = recall = F1); chance level for both = base rate.  Mean over the 3 simulations and 3 topology replicates of each group.
TODO4v2 / BEELINE F1 (undirected, top-k) from heatmap_data_mixed_sweep.csv."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, re
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; H = A + "clean_data/benchmarks/network_benchmarks/score_results/"; pd.set_option("display.width", 250)
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "6000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "rel", "fam"]], on=["raw", "sim", "key"]); Z = Z[Z.fam == "mixed"].dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int)
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
G = pd.DataFrame(rows).groupby("group").mean(); h = pd.read_csv(H + "heatmap_data_mixed_sweep.csv"); h = h[h.variant == "undirected"]
t4 = h[h.method == "TODO4v2"].groupby("group").f1_topk.mean().rename("TODO4v2 F1"); comp = h[h.method != "TODO4v2"].groupby(["group", "method"]).f1_topk.mean().groupby("group").max().rename("best BEELINE F1"); G = G.join(t4).join(comp)
def nm(g): n, e, c, a = map(int, re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)", g).groups()); return f"n{n} e{e} c{c}" + (" autoreg" if a else "")
G["name"] = G.index.map(nm); big = G[G.index.str.startswith("grn_n10")]; small = G[G.index.str.startswith("grn_n6")]
f = lambda v: f"{v:.2f}"
print("n=10 mixed networks, 6,000 twin pairs\n| Group | chance (base) | AUPRC×base \\|z_U\\| | AUPRC×base three-term | F1 \\|z_U\\| | F1 three-term | TODO4v2 F1 | best BEELINE F1 |\n|---|---|---|---|---|---|---|---|")
for g, r in big.iterrows(): print(f"| {r['name']} | {r['base']:.2f} | {f(r['|z_U| AUPRCxbase'])} | {f(r['three-term AUPRCxbase'])} | {f(r['|z_U| F1'])} | {f(r['three-term F1'])} | {f(r['TODO4v2 F1'])} | {f(r['best BEELINE F1'])} |")
cols = [c for c in G.columns if c != "name"]
print("\nmean over groups:"); print(pd.DataFrame({"n=10 (8 groups)": big[cols].mean(), "n=6 (8 groups)": small[cols].mean()}).round(3).to_string())
print("\nn=10: groups where three-term > |z_U|: AUPRC", int((big["three-term AUPRCxbase"] > big["|z_U| AUPRCxbase"]).sum()), "of", len(big), "| F1", int((big["three-term F1"] > big["|z_U| F1"]).sum()))
G.round(3).to_csv(O2 + "rank_f1_auprc_mixed_6000.csv")
