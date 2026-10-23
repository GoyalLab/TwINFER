"""Mixed networks only: score |z_rho| - b|z_C| + g|z_U| by design (X = 1500 twins + 3000 singletons, Y3 = 3000 twins, Y6 = 6000 twins) split by autoregulated vs not.  AUPRC x base, mean over networks."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, re
from sklearn.metrics import average_precision_score
Y = pd.read_csv(A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/twin_design_zscores.csv")
d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]; Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]
lab = d[["raw", "sim", "key", "rel", "fam"]]
X = d[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U"]].assign(design="X 1500tw+3000sing")
Yl = Y.merge(lab, on=["raw", "sim", "key"])[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U", "design"]].assign(design=lambda t: t.design.map({"3000 twins": "Y3 3000 twins", "6000 twins": "Y6 6000 twins"}))
Z = pd.concat([X, Yl], ignore_index=True); Z = Z[Z.fam == "mixed"].dropna(subset=["z_rho", "z_C", "z_U"]); Z["edge"] = Z.rel.str.startswith("edge").astype(int)
ok = Z.groupby(["raw", "sim"]).design.nunique(); Z = Z.set_index(["raw", "sim"]).loc[ok[ok == 3].index].reset_index()
Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs(); Z["auto"] = np.where(Z.raw.str.contains(r"_a[1-9]\d*_pos"), "autoreg", "no autoreg")
G = [(0, 0), (0, 1), (0.5, 1), (1, 1), (1, 0)]; rows = []
for (raw, sim, des), x in Z.groupby(["raw", "sim", "design"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(auto=x.auto.iloc[0], design=des)
    for b, g in G: r[f"b={b},g={g}"] = average_precision_score(x.edge, x.azr - b * x.azC + g * x.azU) / x.edge.mean()
    rows.append(r)
R = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R.groupby(["auto", "design"]).mean(numeric_only=True).round(2).to_string())
