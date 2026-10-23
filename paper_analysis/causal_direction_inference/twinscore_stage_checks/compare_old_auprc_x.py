"""New z-score edge scores vs the OLD figure scores (final_plots/benchmark/synthetic_network_simulations/auprc_x_*; TODO4v2 + BEELINE, twin_paired), same metric:
AUPRC-x = raw AUPRC / base_rate, base_rate = len(GroundTruthNetwork.csv)/(n(n-1)) (self-loops included, as in score_auprc_x_sweeps.py), directed gene pairs,
one AUPRC per (network, sim) replicate, mean over replicates and over the 3 topology replicates of a dataset group.
New scores are pair-level (|z_U|, |z_rho|, scenario score = |z_rho| if |z_C|<=1.96 else |z_U|) and put on directed pairs two ways:
  sym      : the same score for x->y and y->x
  oriented : only the direction the cross-sample delta points to gets the score, the other direction 0 (delta = rho_dag(x->y) - rho_dag(y->x); no ground truth used)
Positives = off-diagonal ground-truth edges.  Output: <O>/compare_old_auprc_x_{replicates,by_group}.csv"""
import re, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; Z = A + "analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores/"
O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; B = A + "code/Beeline/inputs/"
T = 1.96
sets = {"grn_n6_e5_pos100_density": ("network_sweep_final_e5_pairs.csv", "network_sweep_final", "grn_n6_e5_pos100_density_{r}"),
        "grn_n6_e9_pos0_sign_ratio": ("network_sweep_final_e9_pos0_sign_ratio_pairs.csv", "network_sweep_final", "grn_n6_e9_pos0_sign_ratio_{r}"),
        "grn_n6_e9_pos50_sign_ratio": ("network_sweep_final_e9_pos50_sign_ratio_pairs.csv", "network_sweep_final", "grn_n6_e9_pos50_sign_ratio_{r}"),
        "grn_n6_e9_pos100_center": ("network_sweep_final_e9_pos100_center_pairs.csv", "network_sweep_final", "grn_n6_e9_pos100_center_{r}"),
        "n6_e13_pos100": ("network_sweep_final_e13_pos100_pairs_both_designs.csv", "e13_pos100", "n6_e13_pos100_{r}"),
        "grn_n6_e17_pos100_density": ("network_sweep_final_e17_pos100_density_pairs.csv", "network_sweep_final", "grn_n6_e17_pos100_density_{r}")}
frames = []
for g, (f, folder, pat) in sets.items():
    d = pd.read_csv(Z + f); d["raw"] = d.network.map(lambda r: pat.format(r=r)); d["group"] = g; d["gtdir"] = B + folder + "/"; frames.append(d[["raw", "group", "gtdir", "sim", "x", "y", "delta", "z_rho", "z_C", "z_U"]])
m = pd.read_csv(A + "analysis_data/mixed_network_sweep/twinscore_pairs_SCU.csv"); m["raw"] = m.net; m["group"] = m.net.str.replace(r"_rep\d+$", "", regex=True); m["gtdir"] = B + "mixed_network_sweep/"
frames.append(m[["raw", "group", "gtdir", "sim", "x", "y", "delta", "z_rho", "z_C", "z_U"]])
d = pd.concat(frames, ignore_index=True).dropna(subset=["z_rho", "z_C", "z_U", "delta"])
d["s_zU"] = d.z_U.abs(); d["s_zrho"] = d.z_rho.abs(); d["s_scen"] = np.where(d.z_C.abs() > T, d.z_U.abs(), d.z_rho.abs())
SC = {"|z_U|": "s_zU", "|z_rho|": "s_zrho", "scenario (z_rho+PDF)": "s_scen"}
gt_cache = {}
def gt(raw, gtdir):
    if raw not in gt_cache:
        g = pd.read_csv(gtdir + raw + "/GroundTruthNetwork.csv"); n = len(set(g.Gene1) | set(g.Gene2)); off = g[g.Gene1 != g.Gene2]
        gt_cache[raw] = ({(int(a.split("_")[1]), int(b.split("_")[1])) for a, b in zip(off.Gene1, off.Gene2)}, len(g) / (n * (n - 1)))
    return gt_cache[raw]
rows = []
for (raw, sim), x in d.groupby(["raw", "sim"]):
    edges, base = gt(raw, x.gtdir.iloc[0]); xs, ys = x.x.astype(int).to_numpy(), x.y.astype(int).to_numpy(); fw = (x.delta > 0).to_numpy()
    lab = np.r_[[(a, b) in edges for a, b in zip(xs, ys)], [(b, a) in edges for a, b in zip(xs, ys)]].astype(int)
    if lab.sum() == 0: continue
    r = dict(raw=raw, group=x.group.iloc[0], sim=sim)
    for k, s in SC.items():
        v = x[s].to_numpy(); r[k + " sym"] = average_precision_score(lab, np.r_[v, v]) / base
        r[k + " oriented"] = average_precision_score(lab, np.r_[np.where(fw, v, 0), np.where(~fw, v, 0)]) / base
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(O + "compare_old_auprc_x_replicates.csv", index=False)
new = R.groupby("group").mean(numeric_only=True).drop(columns="sim")
def old(fn): o = pd.read_csv(A + "analysis_data/synthetic_network_simulations/" + fn); return o.pivot(index="dataset_id", columns="method", values="auprc_x")
ob = pd.concat([old("auprc_x_network_sweep.csv"), old("auprc_x_mixed_network_sweep.csv")])[["TODO4v2", "PEARSON", "PPCOR", "PIDC", "GRNBOOST2", "GENIE3"]].add_prefix("old ")
T_ = new.join(ob); T_["n_reps"] = R.groupby("group").size(); T_.round(2).to_csv(O + "compare_old_auprc_x_by_group.csv")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(T_.round(2).to_string())
print("\nmean over groups: sweep (6 groups) | mixed (16 groups)")
sw = T_.index.map(lambda s: not s.startswith("grn_n") or "_c" not in s)
print(pd.DataFrame({"sweep": T_[sw].drop(columns="n_reps").mean(), "mixed": T_[~sw].drop(columns="n_reps").mean()}).round(2).to_string())
