"""How much do the X-design results change when X shares cells with the twin-only designs?  Networks: X(old: per-pair random partition) vs X(common: shared permutation) vs N3000 (twin-only); AUPRC x base, mean over groups of each network type.
Two-gene: probability that a random A->B replicate scores above a random A,B replicate, X(old) vs X(common) vs N3000."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; pd.set_option("display.width", 250)
d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Xc = pd.read_csv(O2 + "twin_X_common_networks.csv"); Xc["key"] = [frozenset((int(a), int(b))) for a, b in zip(Xc.x, Xc.y)]
N3 = pd.read_csv(O2 + "twin_decomp_networks.csv"); N3 = N3[N3.N == 3000]; N3["key"] = [frozenset((int(a), int(b))) for a, b in zip(N3.x, N3.y)]
lab = d[["raw", "sim", "key", "rel", "fam"]]; sets = {"X old (per-pair partition)": d[["raw", "sim", "key", "rel", "fam", "z_rho", "z_C", "z_U"]], "X common cells": Xc.merge(lab, on=["raw", "sim", "key"]), "3000 twins": N3.merge(lab, on=["raw", "sim", "key"])}
rows = []
for nm_, Z in sets.items():
    Z = Z.dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int); Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs()
    for (raw, sim), x in Z.groupby(["raw", "sim"]):
        if x.edge.sum() in (0, len(x)): continue
        b = x.edge.mean(); r = dict(design=nm_, group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0])
        for k, s in {"|z_rho|": x.azr, "|z_U|": x.azU, "2 terms": x.azr - x.azC, "3 terms": x.azr - x.azC + x.azU}.items(): r[k] = average_precision_score(x.edge, s) / b
        rows.append(r)
R = pd.DataFrame(rows); R["type"] = np.where(R.group.str.contains("_a[1-9]\\d*_pos"), "mixed autoreg", np.where(R.fam == "mixed", "mixed no autoreg", "sweep"))
G = R.groupby(["design", "type", "group"]).mean(numeric_only=True).reset_index(); T = G.groupby(["design", "type"])[["|z_rho|", "|z_U|", "2 terms", "3 terms"]].mean(); A_ = G.groupby("design")[["|z_rho|", "|z_U|", "2 terms", "3 terms"]].mean(); A_.index = pd.MultiIndex.from_product([A_.index, ["ALL 21+1 groups"]])
print("NETWORKS: AUPRC x base"); print(pd.concat([T, A_]).sort_index().round(3).to_string())
# two-gene
X0 = pd.read_csv(A + "analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores/scenario_table_pictures_figure2.csv")[["scenario", "rep", "z_rho", "z_C", "z_U"]]; X1 = pd.read_csv(O2 + "twin_X_common_twogene.csv")[["scenario", "rep", "z_rho", "z_C", "z_U"]]
N = pd.read_csv(O2 + "twin_decomp_twogene.csv"); N = N[N.N == 3000][["scenario", "rep", "z_rho", "z_C", "z_U"]]; out = []
for nm_, Z in (("X old", X0), ("X common", X1), ("3000 twins", N)):
    Z = Z.copy(); Z["two"] = Z.z_rho.abs() - Z.z_C.abs(); Z["three"] = Z.two + Z.z_U.abs()
    for st in ("single state", "two states", "drift"):
        a = Z[Z.scenario == f"A_B ({st})"]; b = Z[Z.scenario == f"A_to_B ({st})"]; out.append(dict(design=nm_, scenario=st, **{k: round(float(np.mean(b[c].values[:, None] > a[c].values[None, :])), 2) for k, c in (("2 terms", "two"), ("3 terms", "three"))}))
print("\nTWO-GENE: P(random A->B replicate > random A,B replicate), old two/three terms"); print(pd.DataFrame(out).pivot(index="scenario", columns="design", values=["2 terms", "3 terms"]).to_string())
