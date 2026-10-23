"""Separation scored with z_rho_hat_Delta alone (|z_U|) vs the combined score.  (a) figure 2 two-gene scenarios: P(random A->B replicate > random A,B replicate) per state type; (b) figure 3 conditions vs the figure 2 A,B single-state baseline;
(c) calls: fraction of replicates with |z_U| > 1.96 per scenario (A,B rows = false-positive rate).  Scores: |z_U|; exact combined  |z_rho|-|C/sigma_rho|+|U/sigma_rho| (new3); old combined |z_rho|-|z_C|+|z_U| (old3).  Twin-only designs N=1500 and 3000."""
import numpy as np, pandas as pd
O = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; pd.set_option("display.width", 250)
G = pd.read_csv(O + "twin_decomp_twogene.csv"); F3 = pd.read_csv(O + "twin_decomp_figure3.csv"); F3 = F3[F3.design.str.startswith("N")].copy(); F3["N"] = F3.design.str.extract(r"N(\d+)")[0].astype(int)
for T in (G, F3): T["zU"] = T.z_U.abs(); T["new3"] = T.z_rho.abs() - T.A_s.abs() + T.A_u.abs(); T["old3"] = T.z_rho.abs() - T.z_C.abs() + T.z_U.abs()
S = {"|z_U|": "zU", "new combined": "new3", "old combined": "old3"}
P = lambda b, a, c: float(np.mean(b[c].values[:, None] > a[c].values[None, :]))
rows = []
for N in (1500, 3000):
    g = G[G.N == N]
    for st in ("single state", "two states", "drift"):
        a = g[g.scenario == f"A_B ({st})"]; b = g[g.scenario == f"A_to_B ({st})"]; rows.append(dict(N=N, comparison=f"A→B vs A,B ({st})", **{k: round(P(b, a, c), 2) for k, c in S.items()}))
    base = g[g.scenario == "A_B (single state)"]; f = F3[F3.N == N]
    for cond, lab in (("A_rep_B", "A rep. B"), ("A_rep_B_B_rep_A", "A rep. B, B rep. A"), ("A_rep_B_B_to_A", "A rep. B, B → A"), ("A_to_B_B_to_A", "A → B, B → A")):
        rows.append(dict(N=N, comparison=f"fig 3: {lab} vs A,B single", **{k: round(P(f[f.condition == cond], base, c), 2) for k, c in S.items()}))
R = pd.DataFrame(rows); print("P(regulated replicate scores above a no-regulation replicate)"); print(R.pivot(index="comparison", columns="N", values=list(S)).to_string()); R.to_csv(O + "score_zU_vs_combined_separation.csv", index=False)
call = []
for N in (1500, 3000):
    for nm_, T, col in (("fig 2", G[G.N == N], "scenario"), ("fig 3", F3[F3.N == N], "condition")):
        for sc, x in T.groupby(col): call.append(dict(N=N, group=f"{nm_}: {sc}", pct_absz_gt_1p96=round(100 * (x.zU > 1.96).mean()), median_absz=round(x.zU.median(), 2)))
C = pd.DataFrame(call).pivot(index="group", columns="N", values=["pct_absz_gt_1p96", "median_absz"]); print("\n|z_U| > 1.96 (% of replicates) and median |z_U| per scenario"); print(C.to_string())
