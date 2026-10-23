"""Design of an edge ranking from z_U, z_C, z_rho using the PDF scenarios (undirected pairs; same pair set/positives/base as compare_old_auprc_x_undirected.py).
Part 1: enrichment of true edges and of fan-out pairs in each scenario (1.96, no BH).  Part 2: AUPRC x base of tiered / continuous rankings; cross-family check (sweep <-> mixed)."""
import numpy as np, pandas as pd, itertools
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])       # reuse loader + rel() -> d with column 'rel'
T = 1.96
und = {}
def base_pos(raw, gtdir):
    M, R = topo(raw, gtdir); return M
d["azU"] = d.z_U.abs(); d["azC"] = d.z_C.abs(); d["azr"] = d.z_rho.abs()
d["edge"] = d.rel.str.startswith("edge").astype(int); d["fanout"] = d.rel.str.contains("fan-out pair").astype(int)
sh = d.azC > T
d["scen"] = np.where(~sh & (d.azr <= T), 1, np.where(~sh, 2, np.where(d.azU > T, 3, 4)))
print("PART 1 - by scenario (pooled): n, % true edges, % fan-out pairs, enrichment of edges vs overall rate")
for fam, x in [("all", d), ("sweep", d[d.fam == "sweep"]), ("mixed", d[d.fam == "mixed"])]:
    t = x.groupby("scen").agg(n=("edge", "size"), pct_edge=("edge", lambda v: 100 * v.mean()), pct_fanout=("fanout", lambda v: 100 * v.mean())); t["edge_enrichment"] = t.pct_edge / (100 * x.edge.mean()); print(fam, "overall edge rate %.2f" % x.edge.mean()); print(t.round(2).to_string())
def pr(v): return (rankdata(v) - 1) / (len(v) - 1 if len(v) > 1 else 1)          # percent rank in [0,1]
def tier_score(x, h, t=T, within="rho"):
    sh_ = x.azC > t; s2 = ~sh_ & (x.azr > t); s3 = sh_ & (x.azU > t); s4 = sh_ & ~(x.azU > t)
    lev = np.select([s2, s3, s4], [3.0, 2.0, h], 0.0)
    wi = np.where(s3, x.azU, x.azr); return lev + 0.99 * pr(wi)               # tie-break inside a tier: |z_U| for S3, |z_rho| otherwise
def cont_score(x, b, g): return x.azr - b * x.azC + g * x.azU
cands = {"|z_rho|": lambda x: x.azr, "|z_U|": lambda x: x.azU}
for h in (0.0, 1.0, 1.5, 2.0, 2.5): cands[f"tier S2>S3>S4(h={h})>S1"] = (lambda h: lambda x: tier_score(x, h))(h)
for t in (1.0, 1.5): cands[f"tier h=1.5 thr={t}"] = (lambda t: lambda x: tier_score(x, 1.5, t))(t)
for b, g in itertools.product((0, .25, .5, 1.0), (0, .5, 1.0)): 
    if (b, g) != (0, 0): cands[f"|z_rho| - {b}|z_C| + {g}|z_U|"] = (lambda b, g: lambda x: cont_score(x, b, g))(b, g)
rows = []
for (raw, sim), x in d.groupby(["raw", "sim"]):
    if x.edge.sum() == 0: continue
    base = x.edge.mean(); r = dict(raw=raw, fam=x.fam.iloc[0], group=raw.rsplit("_rep", 1)[0])
    for k, f in cands.items(): r[k] = average_precision_score(x.edge, f(x)) / base
    rows.append(r)
R = pd.DataFrame(rows); G = R.groupby(["fam", "group"]).mean(numeric_only=True)
out = pd.DataFrame({"sweep": G.loc["sweep"].mean(), "mixed": G.loc["mixed"].mean()}); out["avg"] = out.mean(axis=1)
print("\nPART 2 - mean AUPRC x base over groups (sweep 6 groups | mixed 16 groups), sorted by avg"); print(out.sort_values("avg", ascending=False).round(3).to_string())
R.to_csv("/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/scenario_ranking_design_replicates.csv", index=False)
