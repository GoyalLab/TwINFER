"""Gated forms: A = U*I + C*(1-I);  B = U + C*I.  I = scenario-3 indicator (|z_C|>1.96 & |z_U|>1.96)  or  I = 1[|z_U|>1.96].  z-versions (all pairs) and raw |U|,|C| versions (not e13: that file has no U/C).
Reference: |z_rho|, previous best |z_rho|-|z_C|+|z_U|, and the old TODO4v2 undirected scores. Metric: undirected AUPRC (raw and x base) per group; same pairs/positives as before."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
T = 1.96
d["azU"] = d.z_U.abs(); d["azC"] = d.z_C.abs(); d["azr"] = d.z_rho.abs(); d["aU"] = d.U.abs(); d["aC"] = d.C.abs() if "C" in d else np.nan; d["edge"] = d.rel.str.startswith("edge").astype(int)
I3 = ((d.azC > T) & (d.azU > T)).astype(float); IU = (d.azU > T).astype(float)
C = {"|z_rho|": d.azr, "prev best |z_rho|-|z_C|+|z_U|": d.azr - d.azC + d.azU,
     "1  |zU|*I3 + |zrho|*(1-I3)": d.azU * I3 + d.azr * (1 - I3), "2  |zrho| + |zU|*I3": d.azr + d.azU * I3,
     "3  |zrho| - |zC|*I3 + |zU|*I3": d.azr - d.azC * I3 + d.azU * I3, "4  |zU| - |zC|*I3": d.azU - d.azC * I3,
     "5  |zU + zC*I3| (signed)": (d.z_U + d.z_C * I3).abs(), "6  |zU*I3 + zC*(1-I3)| (signed A)": (d.z_U * I3 + d.z_C * (1 - I3)).abs(),
     "7  |zrho| - |zC|*(1-I3) + |zU|*I3": d.azr - d.azC * (1 - I3) + d.azU * I3}
for k, v in C.items(): d[k] = v
rows = []
for (raw, sim), x in d.groupby(["raw", "sim"]):
    if x.edge.sum() == 0: continue
    r = dict(raw=raw, group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0], base=x.edge.mean(), k=int(x.edge.sum()))
    for c in C:
        s = x[c]
        if s.isna().any(): r[c + " ap"] = r[c + " topk"] = np.nan; continue
        r[c + " ap"] = average_precision_score(x.edge, s); top = np.argsort(-s.to_numpy(), kind="stable")[:r["k"]]; r[c + " topk"] = x.edge.to_numpy()[top].sum() / r["k"]
    rows.append(r)
R = pd.DataFrame(rows); G = R.groupby(["fam", "group"]).mean(numeric_only=True)
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
o_ratio = pd.read_csv(O2 + "compare_old_undirected_by_group.csv", index_col=0)["old TODO4v2"]; o_raw = pd.read_csv(O2 + "compare_raw_auprc_undirected_by_group.csv", index_col=0)["TODO4v2"]
out = []
for c in C:
    ap = G[c + " ap"]; ratio = ap / G.base; tk = G[c + " topk"] / G.base
    for fam in ("mixed", "sweep"):
        a = ap.loc[fam].dropna(); rt = ratio.loc[fam].reindex(a.index); t = o_ratio.reindex(a.index)
        out.append(dict(score=c, fam=fam, groups=len(a), raw_AUPRC=a.mean(), AUPRC_x_base=rt.mean(), topk_x_base=tk.loc[fam].reindex(a.index).mean(), wins_vs_TODO4v2=f"{int((rt > t).sum())}/{int(t.notna().sum())}" if t.notna().any() else "-", TODO4v2_ratio_same_groups=t.mean()))
O = pd.DataFrame(out); pd.set_option("display.width", 250); print(O.round(3).to_string(index=False))
