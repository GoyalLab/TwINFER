"""Compare designs for the three z-scores: X = 1,500 twins + 3,000 singletons (current, existing pair tables), Y3 = 3,000 twins (6,000 cells), Y6 = 6,000 twins (12,000 cells), all sample B = 20 h, same simulations, same pairs.
Scores: |z_rho|, |z_U|, |z_C|, new3 = |z_rho| - |z_C| + |z_U|.  Metric: undirected AUPRC x base (and raw AUPRC, top-k x base) per group, mean over groups; only (network, sim) present in all designs."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
Y = pd.read_csv(A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/twin_design_zscores.csv")
d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]; Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]
X = d[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U"]].assign(design="X: 1500 twins + 3000 singletons")
Yl = Y.merge(d[["raw", "sim", "key", "rel", "fam"]], on=["raw", "sim", "key"], how="inner")[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U", "design"]]
Yl["design"] = Yl.design.map({"3000 twins": "Y3: 3000 twins (6000 cells)", "6000 twins": "Y6: 6000 twins (12000 cells)"})
A_ = pd.concat([X, Yl], ignore_index=True); A_["edge"] = A_.rel.str.startswith("edge").astype(int)
ok = A_.groupby(["raw", "sim"]).design.nunique(); ok = ok[ok == 3].index; A_ = A_.set_index(["raw", "sim"]).loc[ok].reset_index()
A_["azr"] = A_.z_rho.abs(); A_["azU"] = A_.z_U.abs(); A_["azC"] = A_.z_C.abs(); A_["new3"] = A_.azr - A_.azC + A_.azU
SC = {"|z_rho|": "azr", "|z_U|": "azU", "|z_C|": "azC", "new3": "new3"}; rows = []
for (raw, sim, des), x in A_.groupby(["raw", "sim", "design"]):
    if x.edge.sum() in (0, len(x)): continue
    k = int(x.edge.sum()); r = dict(group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0], design=des, base=x.edge.mean())
    for n, c in SC.items():
        s = x[c].to_numpy(); r[n + " apx"] = average_precision_score(x.edge, s) / r["base"]; r[n + " ap"] = average_precision_score(x.edge, s); r[n + " topk"] = x.edge.to_numpy()[np.argsort(-s, kind="stable")[:k]].sum() / k / r["base"]
    rows.append(r)
R = pd.DataFrame(rows); G = R.groupby(["fam", "group", "design"]).mean(numeric_only=True).reset_index()
pd.set_option("display.width", 250)
for metric, lab in (("apx", "AUPRC x base"), ("ap", "raw AUPRC"), ("topk", "top-k precision x base")):
    cols = [f"{n} {metric}" for n in SC]; T = G.groupby(["fam", "design"])[cols].mean(); T.columns = list(SC); T["groups"] = G.groupby(["fam", "design"]).group.nunique(); print(f"\n{lab} (mean over groups)"); print(T.round(3).to_string())
# paired per-group: Y3 - X
P = G.pivot_table(index=["fam", "group"], columns="design", values=[f"{n} apx" for n in SC]); 
for n in SC:
    dlt = P[f"{n} apx"]["Y3: 3000 twins (6000 cells)"] - P[f"{n} apx"]["X: 1500 twins + 3000 singletons"]; print(f"{n:9s} Y3 - X (AUPRC x base): mean {dlt.mean():+.3f} | groups Y3 better {int((dlt > 0).sum())}/{len(dlt)} | mixed {dlt.loc['mixed'].mean():+.3f} sweep {dlt.loc['sweep'].mean():+.3f}")
G.to_csv(A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/twin_design_compare_by_group.csv", index=False)
