"""Where does the third term matter?  Network pairs (3,000-twin design) stratified by |z_C| (strength of the shared part); AUPRC x base of |z_rho|-|z_C| vs |z_rho|-|z_C|+|z_U| (and |z_U|) pooled within each stratum
(positives = direct-edge pairs, negatives = all other pairs).  Compared with the 2-gene scenarios, whose |z_C| is 18-31 in the 2-state / drift cases.  Output: <O>/two_terms_by_shared_strength.csv"""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "3000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "cls", "fam"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U"]); Z["edge"] = (Z.cls == "direct edge").astype(int)
Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs(); Z["two"] = Z.azr - Z.azC; Z["three"] = Z.two + Z.azU
bins = [(0, 2), (2, 5), (5, 10), (10, 1e9)]; rows = []
for lo, hi in bins:
    x = Z[(Z.azC >= lo) & (Z.azC < hi)]; b = x.edge.mean()
    r = dict(shared_strength=f"|z_C| in [{lo:g}, {hi:g})" if hi < 1e8 else f"|z_C| >= {lo:g}", pairs=len(x), pct_of_pairs=round(100 * len(x) / len(Z), 1), pct_direct_edges=round(100 * b, 1))
    for k, c in (("|z_U|", "azU"), ("2 terms", "two"), ("3 terms", "three")): r[k] = round(average_precision_score(x.edge, x[c]) / b, 2) if 0 < b < 1 else np.nan
    r["3 terms - 2 terms"] = round(r["3 terms"] - r["2 terms"], 2); rows.append(r)
T = pd.DataFrame(rows); pd.set_option("display.width", 250); print(T.to_string(index=False)); T.to_csv(O2 + "two_terms_by_shared_strength.csv", index=False)
print("\nmedian |z_C| in networks: %.2f ; 90th percentile %.2f ; share of pairs with |z_C| > 10: %.1f%%" % (Z.azC.median(), Z.azC.quantile(.9), 100 * (Z.azC > 10).mean()))
x = Z[Z.azC >= 10]; print("\nnetwork pairs with |z_C| >= 10 (n=%d): edges vs non-edges" % len(x)); print(x.groupby("edge")[["azr", "azC", "azU", "two", "three"]].median().round(2).rename(index={0: "non-edge", 1: "direct edge"}).to_string())
print("non-edge pairs with |z_C| >= 10: %d of %d non-edge pairs (%.1f%%); of those, 'co-regulated' %d, 'indirect' %d, 'unrelated' %d" % ((x.edge == 0).sum(), (Z.edge == 0).sum(), 100 * (x.edge == 0).sum() / (Z.edge == 0).sum(), (x.cls == "co-regulated").sum(), (x.cls == "indirect").sum(), (x.cls == "unrelated").sum()))
