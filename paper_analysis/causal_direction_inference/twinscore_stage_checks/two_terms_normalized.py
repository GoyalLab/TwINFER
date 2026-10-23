"""Normalized version of why_two_terms_vs_three.py / two_terms_by_shared_strength.py (3,000-twin design).  Within each panel (one simulation = all its gene pairs) |z_rho|, |z_C|, |z_U| are standardized, s(v) = (v - mean)/sd(ddof=1);
two terms = s_rho - s_C ; three terms = s_rho - s_C + s_U ; 1:.5:.5 = s_rho - 0.5 s_C + 0.5 s_U.   Reports medians by pair class, AUPRC x base for 'direct edge vs <class>', and AUPRC x base by |z_C| stratum.
Output: <O>/two_terms_normalized_*.csv"""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "3000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "cls", "fam"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["edge"] = (Z.cls == "direct edge").astype(int)
Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs()
def s(v): v = np.asarray(v, float); sd = v.std(ddof=1); return (v - v.mean()) / (sd if sd > 0 else 1.0)
for k in ("azr", "azC", "azU"): Z["s_" + k] = Z.groupby(["raw", "sim"])[k].transform(s)
Z["two"] = Z.s_azr - Z.s_azC; Z["three"] = Z.two + Z.s_azU; Z["half"] = Z.s_azr - 0.5 * Z.s_azC + 0.5 * Z.s_azU
pd.set_option("display.width", 250); order = ["direct edge", "indirect", "co-regulated", "unrelated"]
t = Z.groupby("cls")[["s_azr", "s_azC", "s_azU", "two", "three", "half"]].median().reindex(order).round(2); t.columns = ["s(|z_rho|)", "s(|z_C|)", "s(|z_U|)", "2 terms", "3 terms", "1:.5:.5"]; t.insert(0, "n", Z.groupby("cls").size().reindex(order))
print("MEDIANS by pair class (normalized within panel)"); print(t.to_string()); t.to_csv(O2 + "two_terms_normalized_medians.csv")
SC = {"s(|z_rho|)": "s_azr", "s(|z_C|)": "s_azC", "s(|z_U|)": "s_azU", "2 terms": "two", "3 terms": "three", "1:.5:.5": "half"}; rows = []
for neg in ("indirect", "co-regulated", "unrelated", "all non-edges"):
    y = Z[(Z.cls == "direct edge") | ((Z.cls != "direct edge") if neg == "all non-edges" else (Z.cls == neg))]; aps = {k: [] for k in SC}
    for _, z in y.groupby(["raw", "sim"]):
        p = z.edge
        if p.sum() in (0, len(z)): continue
        for k, c in SC.items(): aps[k].append(average_precision_score(p, z[c]) / p.mean())
    rows.append(dict(direct_vs=neg, **{k: round(float(np.mean(v)), 2) for k, v in aps.items()}))
print("\nAUPRC x base, direct edge vs <class>"); print(pd.DataFrame(rows).to_string(index=False)); pd.DataFrame(rows).to_csv(O2 + "two_terms_normalized_auprc.csv", index=False)
rows = []
for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 1e9)]:
    x = Z[(Z.azC >= lo) & (Z.azC < hi)]; b = x.edge.mean(); r = dict(raw_abs_zC=f"[{lo:g}, {hi:g})" if hi < 1e8 else f">= {lo:g}", pairs=len(x), pct_direct=round(100 * b, 1))
    for k in ("s_azU", "two", "three", "half"): r[k] = round(average_precision_score(x.edge, x[k]) / b, 2)
    r["3 - 2"] = round(r["three"] - r["two"], 2); rows.append(r)
print("\nAUPRC x base by raw |z_C| stratum (pooled within stratum)"); print(pd.DataFrame(rows).rename(columns={"s_azU": "s(|z_U|)", "two": "2 terms", "three": "3 terms", "half": "1:.5:.5"}).to_string(index=False))
# overall per-panel AUPRC (mean over panels), for reference
ov = {k: [] for k in SC}
for _, z in Z.groupby(["raw", "sim"]):
    if z.edge.sum() in (0, len(z)): continue
    for k, c in SC.items(): ov[k].append(average_precision_score(z.edge, z[c]) / z.edge.mean())
print("\nmean AUPRC x base over all panels:", {k: round(float(np.mean(v)), 3) for k, v in ov.items()})
