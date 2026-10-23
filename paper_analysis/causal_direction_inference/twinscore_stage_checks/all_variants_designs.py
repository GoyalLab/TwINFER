"""Every score variant (continuous, indicator, scenario-tier, gated) on twin-only designs T1500 (1,500 twin pairs, 3,000 cells) and T3000 (3,000 twin pairs, 6,000 cells), plus the current design X (1,500 twins + 3,000 singletons) as reference.
Same simulations, pairs, positives (direct edge or indirect/etc. -> positive = edge pair, either direction) and metric as before: undirected AUPRC x base per group, mean over groups; top-k precision x base also saved.
Output: <O>/all_variants_designs.csv  (variant x design x network type)."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd, itertools
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; T = 1.96
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]; Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]
X = d[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U"]].assign(design="X 1500 tw + 3000 singletons (ref)")
Yl = Y.merge(d[["raw", "sim", "key", "rel", "fam"]], on=["raw", "sim", "key"])[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U", "design"]]
Yl["design"] = Yl.design.map({"1500 twins": "T1500 1500 twin pairs", "3000 twins": "T3000 3000 twin pairs"}); Yl = Yl.dropna(subset=["design"])
Z = pd.concat([X, Yl], ignore_index=True).dropna(subset=["z_rho", "z_C", "z_U"]); Z["edge"] = Z.rel.str.startswith("edge").astype(int)
ok = Z.groupby(["raw", "sim"]).design.nunique(); Z = Z.set_index(["raw", "sim"]).loc[ok[ok == 3].index].reset_index()
Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs()
def pr(v): v = np.asarray(v, float); return (rankdata(v) - 1) / max(len(v) - 1, 1)
V = {}
V["|z_rho|"] = lambda x: x.azr; V["|z_U|"] = lambda x: x.azU; V["|z_C|"] = lambda x: x.azC
for b in (0, 0.25, 0.5, 1, 1.5):
    for g in (0, 0.5, 1, 2):
        if (b, g) != (0, 0): V[f"cont: |zr| - {b}|zC| + {g}|zU|"] = (lambda b, g: lambda x: x.azr - b * x.azC + g * x.azU)(b, g)
for (b, g) in ((1, 1), (0, 0)):
    for u, c, w in itertools.product((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2)):
        if (b, g) == (0, 0) and (u, c, w) == (0, 0, 0): continue
        if (b, g) == (1, 1) and (u > 2 or c > 1 or w > 1): continue
        V[f"ind: [{b},{g}] + {u}I_U - {c}I_C + {w}I_3"] = (lambda b, g, u, c, w: lambda x: x.azr - b * x.azC + g * x.azU + u * (x.azU > T) - c * (x.azC > T) + w * ((x.azU > T) & (x.azC > T)))(b, g, u, c, w)
def tier(x, h, t=T):
    sh = x.azC > t; s2 = ~sh & (x.azr > t); s3 = sh & (x.azU > t); s4 = sh & ~(x.azU > t); lev = np.select([s2, s3, s4], [3.0, 2.0, h], 0.0); return lev + 0.99 * pr(np.where(s3, x.azU, x.azr))
for h in (0.0, 1.0, 1.5, 2.0, 2.5): V[f"tier: S2>S3>S4(h={h})>S1"] = (lambda h: lambda x: tier(x, h))(h)
I3 = lambda x: ((x.azC > T) & (x.azU > T)).astype(float); IU = lambda x: (x.azU > T).astype(float)
V["gate A: |zU|*I3 + |zC|*(1-I3)"] = lambda x: x.azU * I3(x) + x.azC * (1 - I3(x)); V["gate B: |zU| + |zC|*I3"] = lambda x: x.azU + x.azC * I3(x)
V["gate A': |zU|*IU + |zC|*(1-IU)"] = lambda x: x.azU * IU(x) + x.azC * (1 - IU(x)); V["gate B': |zU| + |zC|*IU"] = lambda x: x.azU + x.azC * IU(x)
V["gate 1: |zU|*I3 + |zr|*(1-I3)"] = lambda x: x.azU * I3(x) + x.azr * (1 - I3(x)); V["gate 2: |zr| + |zU|*I3"] = lambda x: x.azr + x.azU * I3(x)
V["gate 3: |zr| - |zC|*I3 + |zU|*I3"] = lambda x: x.azr - x.azC * I3(x) + x.azU * I3(x); V["gate 4: |zU| - |zC|*I3"] = lambda x: x.azU - x.azC * I3(x)
V["gate 5: |zU + zC*I3|"] = lambda x: (x.z_U + x.z_C * I3(x)).abs(); V["gate 7: |zr| - |zC|*(1-I3) + |zU|*I3"] = lambda x: x.azr - x.azC * (1 - I3(x)) + x.azU * I3(x)
print(len(V), "variants x", Z.design.nunique(), "designs"); rows = []
for (raw, sim, des), x in Z.groupby(["raw", "sim", "design"]):
    if x.edge.sum() in (0, len(x)): continue
    k = int(x.edge.sum()); base = x.edge.mean(); r = dict(group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0], design=des)
    for n, f in V.items():
        s = np.asarray(f(x), float); r[n + "|apx"] = average_precision_score(x.edge, s) / base; r[n + "|topk"] = x.edge.to_numpy()[np.argsort(-s, kind="stable")[:k]].sum() / k / base
    rows.append(r)
R = pd.DataFrame(rows); R["type"] = np.where(R.group.str.contains("_a[1-9]\\d*_pos"), "mixed autoreg", np.where(R.fam == "mixed", "mixed no autoreg", "sweep"))
G = R.groupby(["type", "group", "design"]).mean(numeric_only=True).reset_index()
L = G.melt(id_vars=["type", "group", "design"], var_name="vm", value_name="v"); L[["variant", "metric"]] = L.vm.str.rsplit("|", n=1, expand=True)
S = L.groupby(["variant", "metric", "design", "type"]).v.mean().unstack("type"); S["mixed all"] = (S["mixed autoreg"] + S["mixed no autoreg"]) / 2; S["avg3"] = S[["mixed autoreg", "mixed no autoreg", "sweep"]].mean(axis=1)
S.reset_index().to_csv(O2 + "all_variants_designs.csv", index=False); G.to_csv(O2 + "all_variants_designs_by_group.csv", index=False); print("saved")
