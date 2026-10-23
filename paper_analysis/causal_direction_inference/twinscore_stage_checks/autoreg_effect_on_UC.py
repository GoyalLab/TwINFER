"""How do self-loops (autoregulation) change the pair statistics?  mixed_network_sweep pairs (Figure-2 design, B = 20 h).  Per pair: number of genes with a self-loop (0/1/2) and their sign (from GroundTruthNetwork.csv).
Reports median h (mean of h_x, h_y = sister correlation), |z_rho|, |z_C|, |z_U| for edge and non-edge pairs by self-loop class, and AUROC-free separation as AUPRC x base within each class (positives = edge pairs)."""
import re, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; B = A + "code/Beeline/inputs/mixed_network_sweep/"
m = pd.read_csv(A + "analysis_data/mixed_network_sweep/twinscore_pairs_SCU.csv"); cache = {}
def selfloops(net):
    if net not in cache:
        g = pd.read_csv(B + net + "/GroundTruthNetwork.csv"); s = g[g.Gene1 == g.Gene2]; cache[net] = {int(a.split("_")[1]): t for a, t in zip(s.Gene1, s.Type)}
    return cache[net]
def edges(net):
    g = pd.read_csv(B + net + "/GroundTruthNetwork.csv"); g = g[g.Gene1 != g.Gene2]; return {frozenset((int(a.split("_")[1]), int(b.split("_")[1]))) for a, b in zip(g.Gene1, g.Gene2)}
EC = {}
m["edge"] = [int(frozenset((int(x), int(y))) in EC.setdefault(n, edges(n))) for n, x, y in zip(m.net, m.x, m.y)]
sl = [(selfloops(n).get(int(x)), selfloops(n).get(int(y))) for n, x, y in zip(m.net, m.x, m.y)]
m["n_self"] = [sum(t is not None for t in p) for p in sl]; m["self_pos"] = [sum(t == "+" for t in p) for p in sl]; m["self_neg"] = [sum(t == "-" for t in p) for p in sl]
m = m.dropna(subset=["z_rho", "z_C", "z_U", "h_x", "h_y"]).copy()
m["h"] = (m.h_x.clip(lower=0) + m.h_y.clip(lower=0)) / 2; m["azr"] = m.z_rho.abs(); m["azC"] = m.z_C.abs(); m["azU"] = m.z_U.abs(); m["has_auto_net"] = m.net.str.contains("_a[1-9]\\d*_pos").astype(int)
pd.set_option("display.width", 250); m = m.dropna(subset=["azr", "azC", "azU", "h"])
def tab(x, by):
    g = x.groupby(by); t = g.agg(n=("edge", "size"), pct_edge=("edge", lambda v: 100 * v.mean()), med_h=("h", "median"), med_azr=("azr", "median"), med_azC=("azC", "median"), med_azU=("azU", "median"), pct_zU_sig=("azU", lambda v: 100 * (v > 1.96).mean()), pct_zC_sig=("azC", lambda v: 100 * (v > 1.96).mean()))
    return t.round(2)
print("By number of self-loop genes in the pair (edge vs non-edge):"); print(tab(m, ["n_self", "edge"]).to_string())
print("\nBy self-loop sign (pairs with exactly one self-loop gene):"); one = m[m.n_self == 1].assign(sign=lambda d: np.where(d.self_pos == 1, "+", "-")); print(tab(one, ["sign", "edge"]).to_string())
print("\nSeparation edge vs non-edge, AUPRC x base, within class:")
rows = []
for k, x in m.groupby("n_self"):
    base = x.edge.mean(); rows.append(dict(n_self=k, n=len(x), base=round(base, 2), **{c: round(average_precision_score(x.edge, x[c]) / base, 2) for c in ("azr", "azU", "azC", "h")}, new3=round(average_precision_score(x.edge, x.azr - x.azC + x.azU) / base, 2)))
print(pd.DataFrame(rows).to_string(index=False))
print("\nNetwork level (same n/e/c, with vs without autoregulation), non-edge pairs only:"); ne = m[m.edge == 0].copy(); ne["n_e_c"] = ne.net.str.extract(r"(grn_n\d+_e\d+_c\d+)")[0]
print(ne.groupby(["n_e_c", "has_auto_net"])[["h", "azr", "azC", "azU"]].median().round(2).unstack("has_auto_net").to_string())

# ---- oracle check: is it a scale shift between pair classes?  Rank-normalise the score within self-loop class (n_self), per (net, sim), then pool.
from scipy.stats import rankdata
m["new3"] = m.azr - m.azC + m.azU
def pooled(x, col): return average_precision_score(x.edge, x[col]) / x.edge.mean()
rows = []
for (net, sim), x in m.groupby(["net", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    x = x.copy(); x["new3_cls"] = x.groupby("n_self").new3.transform(lambda v: (rankdata(v) - 1) / max(len(v) - 1, 1)); x["azr_cls"] = x.groupby("n_self").azr.transform(lambda v: (rankdata(v) - 1) / max(len(v) - 1, 1))
    nec = re.match(r"(grn_n\d+_e\d+_c\d+)_a(\d+)", net); rows.append(dict(group=nec.group(1), auto="yes" if int(nec.group(2)) > 0 else "no", new3=pooled(x, "new3"), new3_within_class=pooled(x, "new3_cls"), zrho=pooled(x, "azr"), zrho_within_class=pooled(x, "azr_cls")))
R = pd.DataFrame(rows); print("\nPooled AUPRC x base, raw score vs rank-normalised within self-loop class (oracle):"); print(R.groupby(["group", "auto"]).mean(numeric_only=True).round(2).to_string())
print("\nautoreg networks only, mean over groups:"); print(R[R.auto == "yes"].mean(numeric_only=True).round(3).to_string())

# ---- does the C penalty hurt in autoregulated networks?  |z_rho| - b|z_C| + g|z_U|, split by network type
rows = []
for (net, sim), x in m.groupby(["net", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(auto="autoreg" if re.search(r"_a[1-9]\d*_pos", net) else "no autoreg", n=re.match(r"grn_n(\d+)", net).group(1))
    for b, g in ((0, 0), (0.5, 1), (1, 1), (1, 0), (0, 1)): r[f"b={b},g={g}"] = pooled(x.assign(s=x.azr - b * x.azC + g * x.azU), "s")
    rows.append(r)
Q = pd.DataFrame(rows); print("\nAUPRC x base of |z_rho| - b|z_C| + g|z_U| (mixed, mean over networks):"); print(Q.groupby(["n", "auto"]).mean(numeric_only=True).round(2).to_string())
