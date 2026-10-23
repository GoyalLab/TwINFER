"""Raw (not x base) undirected AUPRC: new score |z_rho| - |z_C| + |z_U| (+ |z_rho|, |z_U|) vs the old heatmap 'undirected' AUPRC of TODO4v2 and BEELINE (twin_paired).
Same pairs/positives as compare_old_auprc_x_undirected.py (positive = unordered gene pair with a ground-truth edge, off-diagonal). Also prints the base rate (= chance AUPRC).
Output: <O>/compare_raw_auprc_undirected_by_group.csv"""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; H = A + "clean_data/benchmarks/network_benchmarks/score_results/"
d["azU"] = d.z_U.abs(); d["azC"] = d.z_C.abs(); d["azr"] = d.z_rho.abs(); d["new"] = d.azr - d.azC + d.azU
d["edge"] = d.rel.str.startswith("edge").astype(int)
rows = []
for (raw, sim), x in d.groupby(["raw", "sim"]):
    if x.edge.sum() == 0: continue
    rows.append(dict(raw=raw, group=raw.rsplit("_rep", 1)[0], base=x.edge.mean(), new=average_precision_score(x.edge, x.new), z_rho=average_precision_score(x.edge, x.azr), z_U=average_precision_score(x.edge, x.azU)))
N = pd.DataFrame(rows).groupby("group").mean(numeric_only=True)
h = pd.concat([pd.read_csv(H + "heatmap_data_network_sweep.csv"), pd.read_csv(H + "heatmap_data_mixed_sweep.csv")]); h = h[h.variant == "undirected"]
h["group"] = h.group.replace({"5 edges": "grn_n6_e5_pos100_density", "9 edges": "grn_n6_e9_pos100_center", "13 edges": "n6_e13_pos100", "all negative": "grn_n6_e9_pos0_sign_ratio", "balanced": "grn_n6_e9_pos50_sign_ratio"})
old = h.groupby(["group", "method"]).auprc.mean().unstack()[["TODO4v2", "PEARSON", "PPCOR", "PIDC", "GRNBOOST2", "GENIE3"]]
T = N.join(old, how="inner"); T.to_csv(O2 + "compare_raw_auprc_undirected_by_group.csv"); T.round(3).to_csv(O2 + "compare_raw_auprc_undirected_by_group.csv")
comp = ["PEARSON", "PPCOR", "PIDC", "GRNBOOST2"]; T["mx"] = T[comp].max(axis=1); T["nb"] = T.new > T.mx; T["tb"] = T.TODO4v2 > T.mx
def nm(g):
    import re; m = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)", g)
    if m: n, e, c, a = map(int, m.groups()); return f"n{n} e{e} c{c}" + (" autoreg" if a else "")
    return {"grn_n6_e5_pos100_density": "sweep e5", "grn_n6_e9_pos0_sign_ratio": "sweep e9, all neg", "grn_n6_e9_pos100_center": "sweep e9, all pos", "grn_n6_e9_pos50_sign_ratio": "sweep e9, balanced", "n6_e13_pos100": "sweep e13"}[g]
fm = lambda v, b: f"**{v:.3f}**" if b else f"{v:.3f}"
print("| Group | base (chance) | new | TODO4v2 | PEARSON | PPCOR | PIDC | GRNBOOST2 | best competitor |\n|---|---|---|---|---|---|---|---|---|")
for g, r in T.sort_index(key=lambda s: s.map(lambda q: (0 if q.startswith("grn_n6_e") and "_c" in q else 1 if "_c" in q else 2, q))).iterrows():
    print(f"| {nm(g)} | {r.base:.3f} | {fm(r.new, r.nb)} | {fm(r.TODO4v2, r.tb)} | " + " | ".join(f"{r[c]:.3f}" for c in comp) + f" | {r.mx:.3f} |")
print(f"\nrows {len(T)} | new > all competitors {int(T.nb.sum())} | TODO4v2 > all competitors {int(T.tb.sum())} | both {int((T.nb & T.tb).sum())} | neither {int((~T.nb & ~T.tb).sum())}")
mixed = T.index.str.contains("_c\\d_a")
for lab, x in (("mixed", T[mixed]), ("sweep", T[~mixed]), ("all", T)): print(lab, len(x), "mean raw AUPRC: new %.3f | |z_rho| %.3f | TODO4v2 %.3f | PEARSON %.3f | PPCOR %.3f | PIDC %.3f | GRNBOOST2 %.3f | base %.3f" % (x.new.mean(), x.z_rho.mean(), x.TODO4v2.mean(), x.PEARSON.mean(), x.PPCOR.mean(), x.PIDC.mean(), x.GRNBOOST2.mean(), x.base.mean()))
