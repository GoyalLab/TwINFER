"""Undirected comparison: new pair-level z-score edge scores vs the OLD 'Unsigned undirected' scores (heatmap_data_{network,mixed}_sweep.csv, variant == undirected: TODO4v2 + BEELINE twin_paired).
Unit = unordered gene pair; positive = pair with a ground-truth edge in either direction (off-diagonal, GroundTruthNetwork.csv); base rate = n_positive_pairs / (n(n-1)/2).
AUPRC-x = raw AUPRC / base. New scores: |z_U|, |z_rho|, scenario score (|z_rho| if |z_C|<=1.96 else |z_U|), one AUPRC per (network, sim); mean over sims and over the 3 topology replicates.
Old scores: raw undirected AUPRC from the heatmap CSVs (one sim per topology replicate) divided by the same per-dataset base.  No orientation is used anywhere.
Output: <O>/compare_old_undirected_{replicates,by_group}.csv"""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; Z = A + "analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores/"
O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; B = A + "code/Beeline/inputs/"; H = A + "clean_data/benchmarks/network_benchmarks/score_results/"
T = 1.96
sets = {"grn_n6_e5_pos100_density": ("network_sweep_final_e5_pairs.csv", "network_sweep_final", "grn_n6_e5_pos100_density_{r}"),
        "grn_n6_e9_pos0_sign_ratio": ("network_sweep_final_e9_pos0_sign_ratio_pairs.csv", "network_sweep_final", "grn_n6_e9_pos0_sign_ratio_{r}"),
        "grn_n6_e9_pos50_sign_ratio": ("network_sweep_final_e9_pos50_sign_ratio_pairs.csv", "network_sweep_final", "grn_n6_e9_pos50_sign_ratio_{r}"),
        "grn_n6_e9_pos100_center": ("network_sweep_final_e9_pos100_center_pairs.csv", "network_sweep_final", "grn_n6_e9_pos100_center_{r}"),
        "n6_e13_pos100": ("network_sweep_final_e13_pos100_pairs_both_designs.csv", "e13_pos100", "n6_e13_pos100_{r}"),
        "grn_n6_e17_pos100_density": ("network_sweep_final_e17_pos100_density_pairs.csv", "network_sweep_final", "grn_n6_e17_pos100_density_{r}")}
frames = []
for g, (f, folder, pat) in sets.items():
    d = pd.read_csv(Z + f); d["raw"] = d.network.map(lambda r: pat.format(r=r)); d["group"] = g; d["gtdir"] = B + folder + "/"; frames.append(d[["raw", "group", "gtdir", "sim", "x", "y", "z_rho", "z_C", "z_U"]])
m = pd.read_csv(A + "analysis_data/mixed_network_sweep/twinscore_pairs_SCU.csv"); m["raw"] = m.net; m["group"] = m.net.str.replace(r"_rep\d+$", "", regex=True); m["gtdir"] = B + "mixed_network_sweep/"
frames.append(m[["raw", "group", "gtdir", "sim", "x", "y", "z_rho", "z_C", "z_U"]])
d = pd.concat(frames, ignore_index=True).dropna(subset=["z_rho", "z_C", "z_U"])
d["|z_U|"] = d.z_U.abs(); d["|z_rho|"] = d.z_rho.abs(); d["scenario (z_rho+PDF)"] = np.where(d.z_C.abs() > T, d.z_U.abs(), d.z_rho.abs())
SC = ["|z_U|", "|z_rho|", "scenario (z_rho+PDF)"]; cache = {}
def gt(raw, gtdir):
    if raw not in cache:
        g = pd.read_csv(gtdir + raw + "/GroundTruthNetwork.csv"); n = len(set(g.Gene1) | set(g.Gene2)); off = g[g.Gene1 != g.Gene2]
        und = {frozenset((int(a.split("_")[1]), int(b.split("_")[1]))) for a, b in zip(off.Gene1, off.Gene2)}; cache[raw] = (und, len(und) / (n * (n - 1) / 2))
    return cache[raw]
rows = []
for (raw, sim), x in d.groupby(["raw", "sim"]):
    und, base = gt(raw, x.gtdir.iloc[0]); lab = np.array([frozenset((int(a), int(b))) in und for a, b in zip(x.x, x.y)]).astype(int)
    if lab.sum() == 0: continue
    rows.append(dict(raw=raw, group=x.group.iloc[0], sim=sim, base=base, **{k: average_precision_score(lab, x[k]) / base for k in SC}))
R = pd.DataFrame(rows); R.to_csv(O + "compare_old_undirected_replicates.csv", index=False)
new = R.groupby("group")[SC].mean()
# old undirected scores from the heatmap data (raw AUPRC / undirected base of that dataset)
h = pd.concat([pd.read_csv(H + "heatmap_data_network_sweep.csv"), pd.read_csv(H + "heatmap_data_mixed_sweep.csv")]); h = h[h.variant == "undirected"]
h = h[~h.group.isin(["5 edges", "9 edges", "13 edges", "all negative", "balanced"]) | True]
gmap = {"5 edges": "grn_n6_e5_pos100_density", "9 edges": "grn_n6_e9_pos100_center", "13 edges": "n6_e13_pos100", "all negative": "grn_n6_e9_pos0_sign_ratio", "balanced": "grn_n6_e9_pos50_sign_ratio"}
h["group"] = h.group.replace(gmap); h["base"] = [gt(r, B + ("e13_pos100/" if r.startswith("n6_e13") else ("network_sweep_final/" if not "_c" in r else "mixed_network_sweep/")))[1] for r in h.dataset_id]
h["x"] = h.auprc / h.base; old = h.groupby(["group", "method"]).x.mean().unstack()[["TODO4v2", "PEARSON", "PPCOR", "PIDC", "GRNBOOST2", "GENIE3"]].add_prefix("old ")
T_ = new.join(old); T_["n_sims_new"] = R.groupby("group").size(); T_ = T_.round(2); T_.to_csv(O + "compare_old_undirected_by_group.csv")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(T_.to_string())
sw = T_.index.map(lambda s: "_c" not in s); S = T_.drop(columns="n_sims_new")
print("\nmean over groups (only groups present in the old heatmaps for the 'old' columns); sweep | mixed"); print(pd.DataFrame({"sweep": S[sw].mean(), "mixed": S[~sw].mean()}).round(2).to_string())
