"""|c z_U| = |U/sigma_rho| vs the three-term scores on the separation tests: P(regulated replicate > no-regulation replicate) for the figure 2 scenarios and the figure 3 conditions (baseline = fig 2 A,B single state), N = 1500 / 3000 twin pairs."""
import numpy as np, pandas as pd
O = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; pd.set_option("display.width", 250)
G = pd.read_csv(O + "twin_decomp_twogene.csv"); F3 = pd.read_csv(O + "twin_decomp_figure3.csv"); F3 = F3[F3.design.str.startswith("N")].copy(); F3["N"] = F3.design.str.extract(r"N(\d+)")[0].astype(int)
for T in (G, F3): T["czU"] = T.A_u.abs(); T["zU"] = T.z_U.abs(); T["new3"] = T.z_rho.abs() - T.A_s.abs() + T.A_u.abs(); T["old3"] = T.z_rho.abs() - T.z_C.abs() + T.z_U.abs()
S = {"|c z_U|": "czU", "|z_U|": "zU", "new 3 terms": "new3", "old 3 terms": "old3"}
P = lambda b, a, c: float(np.mean(b[c].values[:, None] > a[c].values[None, :])); rows = []
for N in (1500, 3000):
    g = G[G.N == N]; base = g[g.scenario == "A_B (single state)"]; f = F3[F3.N == N]
    for st in ("single state", "two states", "drift"):
        a = g[g.scenario == f"A_B ({st})"]; b = g[g.scenario == f"A_to_B ({st})"]; rows.append(dict(N=N, comparison=f"A→B vs A,B ({st})", **{k: round(P(b, a, c), 2) for k, c in S.items()}))
    for cond, lab in (("A_rep_B", "fig 3: A rep. B"), ("A_rep_B_B_rep_A", "fig 3: mutual repression"), ("A_rep_B_B_to_A", "fig 3: A rep. B, B → A"), ("A_to_B_B_to_A", "fig 3: mutual activation")):
        rows.append(dict(N=N, comparison=lab + " vs A,B single", **{k: round(P(f[f.condition == cond], base, c), 2) for k, c in S.items()}))
R = pd.DataFrame(rows); print(R.pivot(index="comparison", columns="N", values=list(S)).to_string()); print("\nmean over the 7 comparisons:"); print(R.groupby("N")[list(S)].mean().round(3).to_string())
