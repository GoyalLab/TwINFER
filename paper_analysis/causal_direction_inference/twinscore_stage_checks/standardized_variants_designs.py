"""Does z-scoring the three statistics before combining help on the network benchmarks?  Per panel (one simulation = all its gene pairs) each of |z_rho|, |z_C|, |z_U| is standardized, s(v) = (v - mean)/sd(ddof=1) (as todo4v2.s() does per panel), then combined:
 s_rho - b*s_C + g*s_U.   Compared with the raw combinations |z_rho| - b|z_C| + g|z_U|.   Designs: X (1500 twins + 3000 singletons), T1500, T3000; metric = AUPRC x base per group, mean over groups of each network type."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]; Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]
X = d[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U"]].assign(design="X 1500 tw + 3000 singletons")
Yl = Y.merge(d[["raw", "sim", "key", "rel", "fam"]], on=["raw", "sim", "key"])[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U", "design"]]; Yl["design"] = Yl.design.map({"1500 twins": "T1500 1500 twin pairs", "3000 twins": "T3000 3000 twin pairs"}); Yl = Yl.dropna(subset=["design"])
Z = pd.concat([X, Yl], ignore_index=True).dropna(subset=["z_rho", "z_C", "z_U"]); Z["edge"] = Z.rel.str.startswith("edge").astype(int)
ok = Z.groupby(["raw", "sim"]).design.nunique(); Z = Z.set_index(["raw", "sim"]).loc[ok[ok == 3].index].reset_index(); Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs()
def s(v): v = np.asarray(v, float); sd = v.std(ddof=1); return (v - v.mean()) / (sd if sd > 0 else 1.0)
V = {"raw |zr|": lambda x: x.azr, "raw |zU|": lambda x: x.azU, "raw |zr|-|zC|+|zU|": lambda x: x.azr - x.azC + x.azU, "raw |zr|-0.5|zC|+|zU|": lambda x: x.azr - 0.5 * x.azC + x.azU}
for b, g in ((0, 0), (0, 1), (0.5, 1), (1, 1), (1, 0), (0.5, 2), (1, 2), (0.5, 0.5), (1, 0.5)):
    if (b, g) == (0, 0): V["std s(zr)"] = lambda x: s(x.azr)
    else: V[f"std s(zr)-{b}s(zC)+{g}s(zU)"] = (lambda b, g: lambda x: s(x.azr) - b * s(x.azC) + g * s(x.azU))(b, g)
V["std s(zU)"] = lambda x: s(x.azU)
rows = []
for (raw, sim, des), x in Z.groupby(["raw", "sim", "design"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0], design=des)
    for n, f in V.items(): r[n] = average_precision_score(x.edge, np.asarray(f(x), float)) / x.edge.mean()
    rows.append(r)
R = pd.DataFrame(rows); R["type"] = np.where(R.group.str.contains("_a[1-9]\\d*_pos"), "mixed autoreg", np.where(R.fam == "mixed", "mixed no autoreg", "sweep"))
G = R.groupby(["type", "group", "design"]).mean(numeric_only=True).reset_index(); T = G.groupby(["design", "type"])[list(V)].mean().T
T.to_csv(O2 + "standardized_variants_designs_by_type.csv"); pd.set_option("display.width", 250)
for d_ in sorted(Z.design.unique()):
    t = T[d_].copy(); t["avg3"] = t.mean(axis=1); print("\n" + d_); print(t.round(3).sort_values("avg3", ascending=False).to_string())
