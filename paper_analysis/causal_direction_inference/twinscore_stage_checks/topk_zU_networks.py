"""Top-k by |z_rho_hat_Delta| (z_U): per (network, sim) rank all gene pairs by |z_U|, take the top k, k = number of true direct-edge pairs of that network.
Reports precision@k (= hits/k = recall@k) and precision@k / base rate (base = k / n_pairs; 1.0 = chance). |z_rho| top-k shown for reference.
Reads analysis_data/.../zrho_pdf_network_auprc/pairs_with_scores.csv (from zrho_pdf_network_auprc.py). Output: topk_zU_per_network.csv, topk_zU_by_group.csv."""
import numpy as np, pandas as pd
O = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
d = pd.read_csv(O + "pairs_with_scores.csv")
def topk(x, s):
    k = int(x.pos.sum()); return x.assign(_s=x[s].fillna(-1)).nlargest(k, "_s", keep="first").pos.sum() / k   # ties: first row; negligible for continuous z
rows = []
for (net, sim), x in d.groupby(["net", "sim"]):
    k = int(x.pos.sum())
    if k == 0 or k == len(x): continue
    rows.append(dict(net=net, sim=sim, family=x.family.iloc[0], group=x.group.iloc[0], n_genes=x.n_genes.iloc[0], edges=x.edges.iloc[0], k=k, n_pairs=len(x), base=k / len(x),
                     p_at_k_zU=topk(x, "s_U"), p_at_k_zrho=topk(x, "s_rho")))
R = pd.DataFrame(rows); R["x_base_zU"] = R.p_at_k_zU / R.base; R["x_base_zrho"] = R.p_at_k_zrho / R.base
N = R.groupby(["net"]).agg(family=("family", "first"), group=("group", "first"), n_genes=("n_genes", "first"), edges=("edges", "first"), sims=("sim", "nunique"), k=("k", "first"), base=("base", "mean"),
                           p_at_k_zU=("p_at_k_zU", "mean"), x_base_zU=("x_base_zU", "mean"), p_at_k_zrho=("p_at_k_zrho", "mean"), x_base_zrho=("x_base_zrho", "mean")).reset_index()
N.to_csv(O + "topk_zU_per_network.csv", index=False)
G = N.groupby(["family", "n_genes", "edges", "group"]).agg(nets=("net", "size"), k=("k", "mean"), base=("base", "mean"), p_at_k_zU=("p_at_k_zU", "mean"), x_base_zU=("x_base_zU", "mean"),
                                                         sd_x_base_zU=("x_base_zU", "std"), p_at_k_zrho=("p_at_k_zrho", "mean"), x_base_zrho=("x_base_zrho", "mean")).round(2)
G.to_csv(O + "topk_zU_by_group.csv"); pd.set_option("display.width", 250); print(G.to_string())
print("\nMIXED by cycle length / autoreg"); m = N[N.family == "mixed"].copy(); m[["c", "a"]] = m.net.str.extract(r"_c(\d+)_a(\d+)_").astype(int).values
print(m.groupby(["n_genes", "edges", "c", "a"])[["p_at_k_zU", "x_base_zU", "x_base_zrho"]].mean().round(2).to_string())
