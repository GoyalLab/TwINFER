"""Figure 3 simulations (4 regulation conditions, 100 replicates each; cache from figure3_decomposition.py): box plots of
  (1) the old metrics  z_rho, z_C, z_U, |z_rho|-|z_C|+|z_U|   for 1500 twins + 3000 singletons (X, built on the shared permutation) vs 3000 twins;
  (2) the exact decomposition  z_rho, C/sigma_rho (= A_s), U/sigma_rho (= A_u = c z_U), |z_rho|-|A_s|+|A_u|   for 1500 vs 3000 twin pairs.
Plus medians of the two-/three-term scores (old and new).   Output: final_plots/.../new_metrics_designs/figure3_*.{pdf,png}"""
import numpy as np, pandas as pd
import plot_new_metrics_boxplots as P
O = P.O; D = pd.read_csv(O + "twin_decomp_figure3.csv")
F3 = ["A_rep_B", "A_rep_B_B_rep_A", "A_rep_B_B_to_A", "A_to_B_B_to_A"]; REF = ["A_B (single state)", "A_to_B (single state)"]      # REF = figure-2 single-state two-gene scenarios (no regulation / A -> B)
CONDS = REF + F3; LAB = ["A, B\n(fig 2)", "A→B\n(fig 2)", "A rep. B", "A rep. B,\nB rep. A", "A rep. B,\nB → A", "A → B,\nB → A"]
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]      # validated categorical slots 1-6
TG = pd.read_csv(O + "twin_decomp_twogene.csv"); TG = TG[TG.scenario.isin(REF)].rename(columns={"scenario": "condition"}); XG = pd.read_csv(O + "twin_X_common_twogene.csv"); XG = XG[XG.scenario.isin(REF)].rename(columns={"scenario": "condition"})
XG["design"] = "X 1500 twins + 3000 singletons"; TG["design"] = TG.N.map({1500: "N1500 twin pairs", 3000: "N3000 twin pairs"})
D = pd.concat([D, XG[["condition", "rep", "design", "z_rho", "z_C", "z_U"]], TG[["condition", "rep", "design", "z_rho", "z_C", "z_U", "A_s", "A_u", "c"]]], ignore_index=True)
# ---- (1) old metrics: X vs N3000
old = D[D.design.isin(["X 1500 twins + 3000 singletons", "N3000 twin pairs"])][["condition", "rep", "design", "z_rho", "z_C", "z_U"]].copy()
old["design"] = old.design.map({"X 1500 twins + 3000 singletons": "1500 twins + 3000 singletons", "N3000 twin pairs": "3000 twins"})
P.make_figure(old, ["1500 twins + 3000 singletons", "3000 twins"], "Figure 3 simulations (+ figure 2 single-state references): 1,500 twins + 3,000 singletons vs 3,000 twins", "figure3_boxplots_old_metrics_designs", ["1500 twins + 3000 singletons", "3000 twins"],
              "\nLeft two groups: figure 2 single-state scenarios (A,B no regulation; A→B regulation); right four: figure 3 regulation conditions. 100 replicates each.\nBoth designs share cells: the 1,500 twin clones of the solid design are the first 1,500 of the 3,000 twins.", order=CONDS, labels=LAB, colors=COLS, scen_col="condition")
# ---- (2) exact decomposition: N1500 vs N3000
nw = D[D.design.isin(["N1500 twin pairs", "N3000 twin pairs"])][["condition", "rep", "design", "z_rho", "A_s", "A_u", "c"]].copy()
nw["design"] = nw.design.map({"N1500 twin pairs": "1500 twin pairs", "N3000 twin pairs": "3000 twin pairs"}); nw["new3"] = nw.z_rho.abs() - nw.A_s.abs() + nw.A_u.abs()
nw["z_U"] = nw.A_u / nw.c; nw["z_C"] = nw.A_s      # [2026-10-17 panel 3 shows z_rho_delta = (U/sigma_rho)/c; the combined score keeps |U/sigma_rho|]
panels = [("z_rho", r"$z_{\rho}$", None, None), ("z_C", r"$C/\sigma_{\rho}$  (shared part, $z_\rho$ units)", None, None), ("z_U", r"$z_{\hat\rho_\Delta} = (U/\sigma_\rho)/c$  (unshared part)", 1.96, None),
          ("new3", r"$|z_\rho| - |C/\sigma_\rho| + |U/\sigma_\rho|$", None, None)]
P.make_figure(nw, ["1500 twin pairs", "3000 twin pairs"], "Figure 3 simulations (+ figure 2 single-state references), exact decomposition z_ρ = C/σ_ρ + U/σ_ρ: 1,500 vs 3,000 twin pairs", "figure3_boxplots_new_decomposition", ["1500 twin pairs", "3000 twin pairs"],
              "\nz_ρ = C/σ_ρ + U/σ_ρ holds exactly for twin-only designs. Panel 3 shows z_ρ̂Δ = (U/σ_ρ)/c; the combined score uses |U/σ_ρ| = c|z_ρ̂Δ|.", order=CONDS, labels=LAB, colors=COLS, panels=panels, scen_col="condition")
# ---- two / three terms
T = D.copy(); T["old2"] = T.z_rho.abs() - T.z_C.abs(); T["old3"] = T.old2 + T.z_U.abs(); T["new2"] = T.z_rho.abs() - T.A_s.abs(); T["new3"] = T.new2 + T.A_u.abs()
pd.set_option("display.width", 250)
print("\nMEDIANS per condition and design (|z_rho|, |z_C|, |z_U|, |A_s|, |A_u|, old2, old3, new2, new3)")
for des, x in T.groupby("design"):
    t = x.groupby("condition")[["z_rho", "z_C", "z_U", "A_s", "A_u", "c", "old2", "old3", "new2", "new3"]].agg(lambda v: v.abs().median() if v.name not in ("old2", "old3", "new2", "new3", "c") else v.median()).loc[CONDS].round(2); print("\n" + des); print(t.to_string())
# ---- figure 2 scenarios (6 two-gene scenarios), exact decomposition, 1500 vs 3000 twin pairs
F2 = pd.read_csv(O + "twin_decomp_twogene.csv"); F2["design"] = F2.N.map({1500: "1500 twin pairs", 3000: "3000 twin pairs"})
F2["new3"] = F2.z_rho.abs() - F2.A_s.abs() + F2.A_u.abs(); F2["z_C"] = F2.A_s; F2["z_U"] = F2.A_u / F2.c      # panel 3 = z_rho_delta (= A_u / c); new3 above uses A_u
ORD2 = ["A_B (single state)", "A_to_B (single state)", "A_B (two states)", "A_to_B (two states)", "A_B (drift)", "A_to_B (drift)"]
P.make_figure(F2, ["1500 twin pairs", "3000 twin pairs"], "Figure 2 scenarios, exact decomposition  z_ρ = C/σ_ρ + U/σ_ρ: 1,500 vs 3,000 twin pairs", "figure2_boxplots_new_decomposition", ["1500 twin pairs", "3000 twin pairs"],
              "\nz_ρ = C/σ_ρ + U/σ_ρ holds exactly for twin-only designs. Panel 3 shows z_ρ̂Δ = (U/σ_ρ)/c; the combined score uses |U/σ_ρ| = c|z_ρ̂Δ|. The 1,500 twin clones are the first 1,500 of the 3,000.", order=ORD2, panels=panels, scen_col="scenario")
# ---- c * z_U (= U / sigma_rho): how the unshared part is rescaled
pc = [("zhd", r"$z_{\hat\rho_\Delta}$", 1.96, None), ("cc", r"$c=\sqrt{(1-h_X)(1-h_Y)}\,\sigma_\Delta/\sigma_\rho$", None, None), ("cz", r"$c\,z_{\hat\rho_\Delta} = U/\sigma_\rho$ (unshared part, $z_\rho$ units)", None, None), ("As", r"$C/\sigma_\rho$ (shared part)", None, None)]
def prep(T, name_col):
    T = T.copy(); T["zhd"] = T.A_u / T.c; T["cc"] = T.c; T["cz"] = T.A_u; T["As"] = T.A_s; T["new3"] = T.A_u; return T
note = "\nz_ρ = C/σ_ρ + c·z_ρ̂Δ. c ≈ √2 when the sister correlations h are small and drops well below 1 as they grow (c = √((1−h_X)(1−h_Y)·2/(1+h_X h_Y)) for twin-only designs); the 1,500 twin clones are the first 1,500 of the 3,000."
f2 = prep(pd.read_csv(O + "twin_decomp_twogene.csv").assign(design=lambda t: t.N.map({1500: "1500 twin pairs", 3000: "3000 twin pairs"})), "scenario")
P.make_figure(f2, ["1500 twin pairs", "3000 twin pairs"], "Figure 2 scenarios: z_ρ̂Δ, the scaling factor c and c·z_ρ̂Δ = U/σ_ρ", "figure2_boxplots_c_times_zU", ["1500 twin pairs", "3000 twin pairs"], note, order=ORD2, panels=pc, scen_col="scenario")
f3 = prep(D[D.design.isin(["N1500 twin pairs", "N3000 twin pairs"])].assign(design=lambda t: t.design.map({"N1500 twin pairs": "1500 twin pairs", "N3000 twin pairs": "3000 twin pairs"})), "condition")
P.make_figure(f3, ["1500 twin pairs", "3000 twin pairs"], "Figure 3 conditions (+ figure 2 single-state references): z_ρ̂Δ, c and c·z_ρ̂Δ = U/σ_ρ", "figure3_boxplots_c_times_zU", ["1500 twin pairs", "3000 twin pairs"], note, order=CONDS, labels=LAB, colors=COLS, panels=pc, scen_col="condition")
