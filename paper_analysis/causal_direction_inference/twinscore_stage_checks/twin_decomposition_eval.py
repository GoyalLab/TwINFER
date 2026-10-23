"""Evaluate the exact decomposition z_rho = A_s + A_u (A_s = C/sigma_rho, A_u = U/sigma_rho = c z_U) against the old two/three-term scores, on networks and on the two-gene scenarios.
 old two = |z_rho|-|z_C|, old three = + |z_U|;  new two = |z_rho|-|A_s|, new three = + |A_u|.   Designs: twin-only N = 1500 and N = 3000 (20 h)."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, os
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; pd.set_option("display.width", 250)
Zn = pd.read_csv(O2 + "twin_decomp_networks.csv"); Zn["key"] = [frozenset((int(a), int(b))) for a, b in zip(Zn.x, Zn.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Zn = Zn.merge(d[["raw", "sim", "key", "cls", "fam"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U", "A_s", "A_u"]).copy(); Zn["edge"] = (Zn.cls == "direct edge").astype(int)
print("identity check z_rho - (A_s + c*z_U): max abs %.2e over %d pairs;  median c = %.2f  (c ~ sqrt(2) when h ~ 0)" % ((Zn.z_rho - Zn.A_s - Zn.c * Zn.z_U).abs().max(), len(Zn), Zn.c.median()))
old = pd.read_csv(O2 + "twin_design_zscores.csv"); old = old[old.design == "3000 twins"].copy(); k = old.merge(Zn[Zn.N == 3000][["raw", "sim", "x", "y", "z_rho"]], on=["raw", "sim", "x", "y"], suffixes=("_old", "_new"))
print("z_rho recomputed vs earlier twin_design_zscores (3000 twins): max abs diff %.2e over %d pairs" % ((k.z_rho_old - k.z_rho_new).abs().max(), len(k)))
for c in ("z_rho", "A_s", "A_u", "z_C", "z_U"): Zn["a_" + c] = Zn[c].abs()
Zn["old2"] = Zn.a_z_rho - Zn.a_z_C; Zn["old3"] = Zn.old2 + Zn.a_z_U; Zn["new2"] = Zn.a_z_rho - Zn.a_A_s; Zn["new3"] = Zn.new2 + Zn.a_A_u
Zn["new_soft"] = Zn.a_z_rho - 0.5 * Zn.a_A_s + Zn.a_A_u
SC = {"|z_rho|": "a_z_rho", "|A_u| (=c|z_U|)": "a_A_u", "|z_U|": "a_z_U", "old 2 terms": "old2", "old 3 terms": "old3", "new 2 terms": "new2", "new 3 terms": "new3", "new soft (b=.5)": "new_soft"}
rows = []
for (N, raw, sim), x in Zn.groupby(["N", "raw", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(N=N, group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0])
    for n, c in SC.items(): r[n] = average_precision_score(x.edge, x[c]) / x.edge.mean()
    rows.append(r)
R = pd.DataFrame(rows); R["type"] = np.where(R.group.str.contains("_a[1-9]\\d*_pos"), "mixed autoreg", np.where(R.fam == "mixed", "mixed no autoreg", "sweep"))
G = R.groupby(["N", "type", "group"]).mean(numeric_only=True).reset_index(); Tn = G.groupby(["N", "type"])[list(SC)].mean(); Tn["all"] = np.nan
for N in (1500, 3000): print(f"\nNETWORKS, N = {N} twin pairs: AUPRC x base (mean over groups of each type)"); t = Tn.loc[N].drop(columns="all").T; t["avg3"] = t.mean(axis=1); t["all 21 groups"] = G[G.N == N][list(SC)].mean(); print(t.round(3).to_string())
Tn.round(4).to_csv(O2 + "twin_decomp_eval_networks.csv")
Z3 = Zn[Zn.N == 3000]; order = ["direct edge", "indirect", "co-regulated", "unrelated"]
t = Z3.groupby("cls")[["a_z_rho", "a_z_C", "a_A_s", "a_z_U", "a_A_u", "old2", "new2", "old3", "new3"]].median().reindex(order).round(2); print("\nMEDIANS by pair class, N = 3000"); print(t.to_string())
for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 1e9)]:
    x = Z3[(Z3.a_A_s >= lo) & (Z3.a_A_s < hi)]; b = x.edge.mean(); print(f"|A_s| in [{lo:g},{hi:g}): pairs {len(x)}, direct {100*b:.0f}% | new2 {average_precision_score(x.edge, x.new2)/b:.2f}  new3 {average_precision_score(x.edge, x.new3)/b:.2f}  old2 {average_precision_score(x.edge, x.old2)/b:.2f}  old3 {average_precision_score(x.edge, x.old3)/b:.2f}")
