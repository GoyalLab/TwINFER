"""Control experiment: 4 autoregulated mixed networks simulated WITHOUT their self-loops (simulation_data/mixed_network_noself, 1 replicate each) vs the same networks WITH self-loops (original sims 0-2).
Pair statistics computed exactly as for the originals (mixed_network_sweep_pairs.py: Figure-2 design, F.analyse, B = 20 h).  Reports gene-level h of the (original) self-loop genes, pair statistics by edge / non-edge, and scores (AUPRC x base, top-k x base)."""
import glob, os, re, sys, numpy as np, pandas as pd
sys.argv_backup = sys.argv; sys.argv = [sys.argv[0]]
import figure2_1000_zscores as F
from network_sweep_scenarios import classes
sys.argv = sys.argv_backup
from sklearn.metrics import average_precision_score
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; SIMN = A + "simulation_data/mixed_network_noself/"
def one(f):
    m = re.search(r"df_(grn_.+?_rep\d+)_rep(\d+)_\d{8}_", os.path.basename(f)); net, sim = m.group(1), int(m.group(2))
    C = np.loadtxt(A + f"input_data/mixed_network_sweep/{net}.txt", delimiter=",").astype(int); n = C.shape[0]; C0 = C.copy(); np.fill_diagonal(C0, 0); cl = classes(C0)
    d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate"] + [f"gene_{k}_mRNA" for k in range(1, n + 1)])
    P = {t: {k: d[d.time_step == t].pivot(index="clone_id", columns="replicate", values=f"gene_{k}_mRNA").values for k in range(1, n + 1)} for t in (1, 20)}; rows = []
    for (i, j), (c, x, y) in cl.items():
        gx, gy = x + 1, y + 1; rng = np.random.RandomState(7 * sim + 100 * i + j)
        if P[20][gx].std() == 0 or P[20][gy].std() == 0: continue
        o = F.analyse({"gene_1": P[1][gx], "gene_2": P[1][gy]}, {"gene_1": P[20][gx], "gene_2": P[20][gy]}, rng, 200); o.update(net=net, sim=sim, n_genes=n, pair=f"{i+1}-{j+1}", cls=c, x=gx, y=gy); rows.append(o)
    return rows
out = O + "control_noself_pairs.csv"
if not os.path.exists(out):
    fs = sorted(f for f in glob.glob(SIMN + "df_grn_*.csv")); print(len(fs), "noself files"); pd.DataFrame([r for f in fs for r in one(f)]).to_csv(out, index=False)
N = pd.read_csv(out); N["self"] = "no self-loops"
W = pd.read_csv(A + "analysis_data/mixed_network_sweep/twinscore_pairs_SCU.csv"); W = W[W.net.isin(N.net.unique()) & (W.sim <= 2)].copy(); W["self"] = "with self-loops"
D = pd.concat([W, N], ignore_index=True).dropna(subset=["z_rho", "z_C", "z_U"]); D["edge"] = (D.cls == "direct edge").astype(int); D["azr"] = D.z_rho.abs(); D["azC"] = D.z_C.abs(); D["azU"] = D.z_U.abs()
D["short"] = D.net.str.extract(r"grn_(n\d+_e\d+_c\d+)")[0]; pd.set_option("display.width", 220)
# gene-level h of the original self-loop genes
sl = {}
for net in N.net.unique():
    g = pd.read_csv(A + f"code/Beeline/inputs/mixed_network_sweep/{net}/GroundTruthNetwork.csv"); sl[net] = {int(a.split("_")[1]): t for a, t, b in zip(g.Gene1, g.Type, g.Gene2) if a == b}
G = pd.concat([D[["net", "sim", "self", "x", "h_x"]].rename(columns={"x": "gene", "h_x": "h"}), D[["net", "sim", "self", "y", "h_y"]].rename(columns={"y": "gene", "h_y": "h"})]).groupby(["net", "self", "sim", "gene"]).h.mean().reset_index()
G["loop"] = [sl[n].get(int(g), "none") for n, g in zip(G.net, G.gene)]; G["h"] = G.h.clip(lower=0)
print("\n1. sister correlation h by gene type (mean over genes; original self-loop genes labelled by sign):"); print(G.groupby(["loop", "self"]).h.agg(["mean", "median", "size"]).round(3).unstack("self").to_string())
print("\n2. pair statistics (all 4 networks pooled), edge vs non-edge:")
t = D.groupby(["edge", "self"]).agg(n=("azr", "size"), med_zrho=("azr", "median"), med_zC=("azC", "median"), med_zU=("azU", "median"), pct_zC=("azC", lambda v: 100 * (v > 1.96).mean()), pct_zU=("azU", lambda v: 100 * (v > 1.96).mean())).round(2); print(t.to_string())
print("\n   by network:"); t = D.groupby(["short", "edge", "self"]).agg(n=("azr", "size"), med_zrho=("azr", "median"), med_zC=("azC", "median"), med_zU=("azU", "median"), pct_zU=("azU", lambda v: 100 * (v > 1.96).mean())).round(2); print(t.to_string())
def score(x, b, g): return x.azr - b * x.azC + g * x.azU
SC = {"|z_rho|": (0, 0), "|z_U|": None, "|z_rho|+|z_U| (b=0)": (0, 1), "soft b=0.5": (0.5, 1), "full b=1": (1, 1), "|z_rho|-|z_C|": (1, 0)}; rows = []
for (net, sim, s), x in D.groupby(["net", "sim", "self"]):
    if x.edge.sum() in (0, len(x)): continue
    k = int(x.edge.sum()); r = dict(short=x.short.iloc[0], self=s, sim=sim)
    for n_, p in SC.items():
        v = x.azU if p is None else score(x, *p); r[n_] = average_precision_score(x.edge, v) / x.edge.mean(); r[n_ + " topk"] = x.edge.to_numpy()[np.argsort(-v.to_numpy(), kind="stable")[:k]].sum() / k / x.edge.mean()
    rows.append(r)
R = pd.DataFrame(rows); print("\n3. AUPRC x base per network (mean of sims 0-2 [min-max] in both conditions):")
for short, x in R.groupby("short"):
    print(short); w = x[x.self == "with self-loops"]; n_ = x[x.self == "no self-loops"]
    print(pd.DataFrame({"with self-loops": [f"{w[c].mean():.2f} [{w[c].min():.2f}-{w[c].max():.2f}]" for c in SC], "no self-loops": [f"{n_[c].mean():.2f} [{n_[c].min():.2f}-{n_[c].max():.2f}]" for c in SC]}, index=list(SC)).to_string())
print("\n4. mean over the 4 networks (AUPRC x base | top-k x base):"); M = R.groupby(["self"])[[c for c in R.columns if c not in ("short", "self", "sim")]].mean().T; M = M.round(2); print(M.to_string())
print("\n   benefit of the C penalty (b=1 minus |z_rho|+|z_U|, b=0) by network and condition:"); R["C_penalty_gain"] = R["full b=1"] - R["|z_rho|+|z_U| (b=0)"]; print(R.groupby(["short", "self"]).C_penalty_gain.mean().round(2).unstack("self").to_string())
R.to_csv(O + "control_noself_scores.csv", index=False)
