"""AUPRC x base rate (1.0 = chance) of the PDF (15 Oct) scenario rule with z_rho deciding 1 vs 2, on every network_sweep_final set and every mixed_network_sweep network,
per network, then size/topology-matched groups.  Positive = direct-edge gene pair (either direction); negatives = all other pairs of the same network (all sims pooled per network).
Scores (all from the existing per-pair z-scores, Figure-2 design, B = 20 h, mRNA only, no new simulation):
  |z_rho|                 co-expression baseline
  |z_U|                   twin-difference test alone (PDF scenario 2/3 axis)
  |z_C|                   shared-state test alone
  scenario_score          z_rho + PDF rule: |z_rho| if shared part not significant (|z_C|<=1.96, scenarios 1/2) else |z_U| (scenarios 3/4)
  call S2|S3 (binary)     scenario 2 or 3 called at |z|>1.96, no BH
usage: python zrho_pdf_network_auprc.py"""
import glob, re, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/"
Z = A + "paper_analysis/causal_direction_inference/figure2_1000_zscores/"
OUT = A + "paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
T = 1.96
sets = {"e5_pos100": "network_sweep_final_e5_pairs.csv", "e9_pos0": "network_sweep_final_e9_pos0_sign_ratio_pairs.csv",
        "e9_pos50": "network_sweep_final_e9_pos50_sign_ratio_pairs.csv", "e9_pos100": "network_sweep_final_e9_pos100_center_pairs.csv",
        "e13_pos100": "network_sweep_final_e13_pos100_pairs_both_designs.csv", "e17_pos100": "network_sweep_final_e17_pos100_density_pairs.csv"}
rows = []
for k, f in sets.items():
    d = pd.read_csv(Z + f); d["net"] = f"sweep_{k}_" + d.network.astype(str); d["family"] = "network_sweep_final"; d["group"] = k
    d["n_genes"] = 6; d["edges"] = int(re.match(r"e(\d+)", k).group(1)); rows.append(d[["net", "family", "group", "n_genes", "edges", "sim", "cls", "z_rho", "z_C", "z_U"]])
m = pd.read_csv(A + "mixed_network_sweep/twinscore_pairs_SCU.csv")
mm = re.compile(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)_pos50_rep(\d+)")
g = m.net.map(lambda s: mm.match(s).groups()).apply(pd.Series).astype(int); g.columns = ["n", "e", "c", "a", "r"]
m = m.join(g); m["family"] = "mixed"; m["edges"] = m.e; m["group"] = "mixed_n" + m.n.astype(str) + "_e" + m.e.astype(str)
rows.append(m[["net", "family", "group", "n_genes", "edges", "sim", "cls", "z_rho", "z_C", "z_U", "c", "a"]])
d = pd.concat(rows, ignore_index=True).dropna(subset=["z_rho", "z_C", "z_U"])
d["pos"] = d.cls == "direct edge"
sh = d.z_C.abs() > T
d["s_rho"] = d.z_rho.abs(); d["s_U"] = d.z_U.abs(); d["s_C"] = d.z_C.abs()
d["s_scen"] = np.where(~sh, d.z_rho.abs(), d.z_U.abs())
d["s_call"] = np.where(~sh, d.z_rho.abs() > T, d.z_U.abs() > T).astype(float)    # scenario 2 (shared ns, z_rho sig) or 3 (shared sig, z_U sig)
SC = {"|z_rho|": "s_rho", "|z_U|": "s_U", "|z_C|": "s_C", "scenario_score (z_rho + PDF)": "s_scen", "call S2|S3 (binary)": "s_call"}
def ap(x, s):
    b = x.pos.mean(); return average_precision_score(x.pos, x[s]) / b if 0 < b < 1 else np.nan
res = []
for net, x in d.groupby("net"):
    r = dict(net=net, family=x.family.iloc[0], group=x.group.iloc[0], n_genes=x.n_genes.iloc[0], edges=x.edges.iloc[0], n_pairs=len(x), n_pos=int(x.pos.sum()), base=x.pos.mean())
    for k, s in SC.items(): r[k] = ap(x, s)
    for c in ("c", "a"):
        if c in x: r[c] = x[c].iloc[0]
    res.append(r)
R = pd.DataFrame(res); R.to_csv(OUT + "auprc_x_base_per_network.csv", index=False)
cols = list(SC)
G = R.groupby(["family", "n_genes", "edges", "group"])[cols].agg(["mean", "std"]).round(2); G["n_networks"] = R.groupby(["family", "n_genes", "edges", "group"]).size()
G.to_csv(OUT + "auprc_x_base_by_group.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("AUPRC x base rate per group (mean over networks of the group; 1.0 = chance)")
print(R.groupby(["family", "n_genes", "edges", "group"])[cols].mean().round(2).assign(n_nets=R.groupby(["family", "n_genes", "edges", "group"]).size()).to_string())
mx = R[R.family == "mixed"]
print("\nMIXED by cycle length / autoregulation (mean over networks)")
print(mx.groupby(["n_genes", "edges", "c", "a"])[cols].mean().round(2).to_string())
# size/topology matched: n=6 edge counts 5<->6 and 13<->12 (closest), pooled-pair AUPRC x base per group
mt = {"mixed n6 e6 vs sweep e5": (["mixed_n6_e6"], ["e5_pos100"]), "mixed n6 e12 vs sweep e13": (["mixed_n6_e12"], ["e13_pos100"])}
print("\nMATCHED (n=6, closest edge count), mean of per-network AUPRC x base [per-network sd]")
for k, (a, b) in mt.items():
    for lab, gs in (("mixed", a), ("sweep", b)):
        x = R[R.group.isin(gs)]; print(f"{k:28s} {lab:6s} nets={len(x):2d}  " + "  ".join(f"{c.split(' ')[0]}={x[c].mean():.2f}[{x[c].std():.2f}]" for c in cols))
d.to_csv(OUT + "pairs_with_scores.csv", index=False)
