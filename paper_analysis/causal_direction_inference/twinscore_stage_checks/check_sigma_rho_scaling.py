"""Does it make sense to divide C and U by sigma_rho?  (1) null spread: SD over A,B single-state replicates (no regulation) of z_rho, C/sigma_rho, U/sigma_rho, z_C, z_U.  (2) algebra: new3 = |z_rho|-|A_s|+|A_u| equals 2|A_u| when A_s and A_u have the same sign, 2(|A_u|-|A_s|) or 0 when they oppose;
how often signs agree for direct edges vs non-edges.  (3) network AUPRC x base for alternative scalings (3,000 and 1,500 twin pairs)."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; pd.set_option("display.width", 250)
G = pd.read_csv(O2 + "twin_decomp_twogene.csv"); ab = G[(G.scenario == "A_B (single state)")]
print("(1) NULL SPREAD, A,B single state (no regulation), SD over 100 replicates")
print(ab.groupby("N")[["z_rho", "A_s", "A_u", "z_C", "z_U"]].std().round(2).rename(columns={"A_s": "C/sigma_rho", "A_u": "U/sigma_rho"}).to_string()); print("   median c =", ab.c.median().round(2))
Z = pd.read_csv(O2 + "twin_decomp_networks.csv"); Z["key"] = [frozenset((int(a), int(b))) for a, b in zip(Z.x, Z.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Z.merge(d[["raw", "sim", "key", "rel", "fam"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U", "A_s", "A_u"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int)
Z["same"] = np.sign(Z.A_s) == np.sign(Z.A_u); Z["new3"] = Z.z_rho.abs() - Z.A_s.abs() + Z.A_u.abs()
z3 = Z[Z.N == 3000]; print("\n(2) sign agreement of shared and unshared parts (N=3000), share of pairs with sign(C) = sign(U):"); print(z3.groupby("edge").same.mean().round(2).rename(index={0: "non-edges", 1: "direct edges"}).to_string())
print("   new3 vs 2|A_u| when signs agree: max abs diff %.2e" % (z3[z3.same].new3 - 2 * z3[z3.same].A_u.abs()).abs().max())
az = lambda c: Z[c].abs()
V = {"new3 = |zr|-|A_s|+|A_u|": Z.new3, "|A_u| alone (=c|z_U|)": az("A_u"), "|z_U| alone": az("z_U"), "new2 = |zr|-|A_s|": az("z_rho") - az("A_s"),
     "U scaled by own SD: |zr|-|A_s|+|z_U|": az("z_rho") - az("A_s") + az("z_U"), "C scaled by own SD: |zr|-|z_C|+|A_u|": az("z_rho") - az("z_C") + az("A_u"), "old: |zr|-|z_C|+|z_U|": az("z_rho") - az("z_C") + az("z_U"),
     "sign-agreement only: |A_u|*same": az("A_u") * Z.same, "|z_U|*same": az("z_U") * Z.same}
rows = []
for (N, raw, sim), x in Z.groupby(["N", "raw", "sim"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(N=N, group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0])
    for k, v in V.items(): r[k] = average_precision_score(x.edge, v.loc[x.index]) / x.edge.mean()
    rows.append(r)
R = pd.DataFrame(rows); R["type"] = np.where(R.group.str.contains("_a[1-9]\\d*_pos"), "mixed autoreg", np.where(R.fam == "mixed", "mixed no autoreg", "sweep"))
Gm = R.groupby(["N", "type", "group"]).mean(numeric_only=True).reset_index()
for N in (3000, 1500):
    t = Gm[Gm.N == N].groupby("type")[list(V)].mean().T; t["all 21"] = Gm[Gm.N == N][list(V)].mean(); print(f"\n(3) AUPRC x base, N = {N} twin pairs (mean over groups of each type)"); print(t.round(3).to_string())
