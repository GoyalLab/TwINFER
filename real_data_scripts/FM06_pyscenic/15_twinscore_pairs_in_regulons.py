# FM06: how many TwinScore TF-target pairs fall inside a SCENIC regulon, and how that relates to the three TwinScore z-scores.
# TwinScore pairs: analysis_data/fm06/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv (ChIP-Atlas Epidermis TF x target candidates, PDF procedure, 140,814 pairs; Stage I = 39,320)
#   z-scores: z_rho (Stage I), h = sign(S) C / sigma_C ("z_C", inherited state), z_U (Stage III); absolute values used, a regulon has no sign
# SCENIC: analysis_data/fm06/pyscenic/compare/all_edges_FM06_GRNBoost2_U_S.csv (regulon membership per TF-target edge, self-edges removed):
#   in_GRNBoost2_regulon = standard SCENIC (GRNBoost2 adjacencies -> cisTarget motif pruning -> regulon); in_U_regulon / in_S_regulon = same pipeline with TwINFER U or S in place of GRNBoost2 weights
# A pair "falls within a regulon" when its target is in the regulon of its TF. Only the 30 of 43 candidate TFs that have a regulon can contribute, so every rate below is also given within those TFs.
import os, numpy as np, pandas as pd
from scipy.stats import fisher_exact, spearmanr
R = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/"; OUT = R + "pyscenic/compare/twinscore_pairs_vs_regulons/"; os.makedirs(OUT, exist_ok=True)
E = pd.read_csv(R + "pyscenic/compare/all_edges_FM06_GRNBoost2_U_S.csv")
A = pd.read_csv(R + "data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv")
E = E[["TF", "target", "in_GRNBoost2_regulon", "in_U_regulon", "in_S_regulon", "GRNBoost2_importance", "in_CollecTRI"]].drop_duplicates(["TF", "target"])
M = A.merge(E, left_on=["TF", "gene"], right_on=["TF", "target"], how="left").drop(columns="target")
for c in ["in_GRNBoost2_regulon", "in_U_regulon", "in_S_regulon", "in_CollecTRI"]: M[c] = M[c].eq(True)
M["any_regulon"] = M[["in_GRNBoost2_regulon", "in_U_regulon", "in_S_regulon"]].any(axis=1); M["all_three"] = M[["in_GRNBoost2_regulon", "in_U_regulon", "in_S_regulon"]].all(axis=1)
for z in ("z_rho", "h", "z_U"): M["abs_" + z] = M[z].abs()
reg_tf = {m: set(M.loc[M[m], "TF"]) for m in ("in_GRNBoost2_regulon", "in_U_regulon", "in_S_regulon", "any_regulon")}
M.to_csv(OUT + "fm06_twinscore_pairs_with_regulon_flags.csv.gz", index=False)
meth = ["in_GRNBoost2_regulon", "in_U_regulon", "in_S_regulon", "any_regulon", "all_three"]; nm = {"in_GRNBoost2_regulon": "GRNBoost2 SCENIC", "in_U_regulon": "U-weighted", "in_S_regulon": "S-weighted", "any_regulon": "any of the three", "all_three": "all three"}
sets = {"candidate pairs (all)": M, "Stage I pairs (no gene filter)": M[M.stage1], "listed pairs (Stage I, gene filter)": M[M.stage1 & ~M.filtered_out], "called, q_U <= 0.05": M[M.called], "inherited state (h > 2.326)": M[M.inherited]}
rows = []
for k, d in sets.items():
    r = {"set": k, "pairs": len(d)}
    for m in meth: r[nm[m]] = int(d[m].sum()); r[nm[m] + " %"] = round(100 * d[m].mean(), 2)
    rows.append(r)
T1 = pd.DataFrame(rows); T1.to_csv(OUT + "table1_pairs_in_regulon_by_set.csv", index=False)
# same, restricted to TFs that have a regulon (GRNBoost2 / any)
rows = []
for k, d in sets.items():
    r = {"set": k}
    for m in ("in_GRNBoost2_regulon", "any_regulon"):
        dd = d[d.TF.isin(reg_tf[m])]; r[f"pairs of TFs with a {nm[m]} regulon"] = len(dd); r[f"in regulon"] = int(dd[m].sum()); r["%"] = round(100 * dd[m].mean(), 2)
        rows.append({**r, "regulon": nm[m]}); r = {"set": k}
T2 = pd.DataFrame(rows)[["set", "regulon", "pairs of TFs with a GRNBoost2 SCENIC regulon", "pairs of TFs with a any of the three regulon", "in regulon", "%"]] if False else pd.DataFrame(rows)
T2.to_csv(OUT + "table2_pairs_in_regulon_TFs_with_regulon.csv", index=False)
# top-N by each z among Stage I pairs of TFs with a GRNBoost2 regulon
S1 = M[M.stage1 & M.TF.isin(reg_tf["in_GRNBoost2_regulon"])]; rows = []
for z in ("abs_z_rho", "abs_h", "abs_z_U"):
    for n in (100, 500, 1000, 3000):
        d = S1.nlargest(n, z); rows.append({"ranked_by": z.replace("abs_", "|") + "|", "top_N": n, "GRNBoost2 regulon": int(d.in_GRNBoost2_regulon.sum()), "%GRNBoost2": round(100 * d.in_GRNBoost2_regulon.mean(), 1),
                                            "%U-weighted": round(100 * d.in_U_regulon.mean(), 1), "%S-weighted": round(100 * d.in_S_regulon.mean(), 1), "%any": round(100 * d.any_regulon.mean(), 1), "%all three": round(100 * d.all_three.mean(), 1)})
rows.append({"ranked_by": "all Stage I pairs of these TFs", "top_N": len(S1), "GRNBoost2 regulon": int(S1.in_GRNBoost2_regulon.sum()), "%GRNBoost2": round(100 * S1.in_GRNBoost2_regulon.mean(), 1), "%U-weighted": round(100 * S1.in_U_regulon.mean(), 1), "%S-weighted": round(100 * S1.in_S_regulon.mean(), 1), "%any": round(100 * S1.any_regulon.mean(), 1), "%all three": round(100 * S1.all_three.mean(), 1)})
T3 = pd.DataFrame(rows); T3.to_csv(OUT + "table3_top_by_z_in_regulon.csv", index=False)
# association of regulon membership with each z (Stage I pairs of TFs with a GRNBoost2 regulon): AUC of |z| for in-regulon vs not, and rho with GRNBoost2 importance
from scipy.stats import mannwhitneyu
rows = []
for z in ("abs_z_rho", "abs_h", "abs_z_U"):
    for m in ("in_GRNBoost2_regulon", "in_U_regulon", "in_S_regulon"):
        a, b = S1.loc[S1[m], z], S1.loc[~S1[m], z]; auc = mannwhitneyu(a, b).statistic / (len(a) * len(b))
        rows.append({"z": z.replace("abs_", "|") + "|", "regulon": nm[m], "n_in": len(a), "n_out": len(b), "median_in": round(a.median(), 2), "median_out": round(b.median(), 2), "AUC_in_vs_out": round(auc, 3)})
T4 = pd.DataFrame(rows); T4.to_csv(OUT + "table4_z_in_vs_out_of_regulon.csv", index=False)
ii = S1[S1.in_GRNBoost2_regulon & S1.GRNBoost2_importance.notna()]
rows = [{"z": z, "n": len(ii), "spearman_with_GRNBoost2_importance": round(spearmanr(ii[z], ii.GRNBoost2_importance)[0], 3)} for z in ("abs_z_rho", "abs_h", "abs_z_U")]
T5 = pd.DataFrame(rows); T5.to_csv(OUT + "table5_z_vs_grnboost_importance.csv", index=False)
# per TF
g = S1.groupby("TF").agg(stage1_pairs=("gene", "size"), in_GRNBoost2=("in_GRNBoost2_regulon", "sum"), in_any=("any_regulon", "sum"), called=("called", "sum"), inherited=("inherited", "sum")).reset_index()
g["called_in_GRNBoost2"] = [int(S1[(S1.TF == t) & S1.called].in_GRNBoost2_regulon.sum()) for t in g.TF]
g["pct_stage1_in_GRNBoost2"] = (100 * g.in_GRNBoost2 / g.stage1_pairs).round(1); g["pct_called_in_GRNBoost2"] = (100 * g.called_in_GRNBoost2 / g.called.replace(0, np.nan)).round(1)
g = g.sort_values("in_GRNBoost2", ascending=False); g.to_csv(OUT + "table6_per_TF.csv", index=False)
# pairs that belong most to SCENIC: in all three regulons, ranked by the largest of the three |z|; with the GRNBoost2 importance
P = M[M.stage1 & M.all_three].copy(); P["max_abs_z"] = P[["abs_z_rho", "abs_h", "abs_z_U"]].max(axis=1)
P = P.sort_values("max_abs_z", ascending=False)[["TF", "gene", "z_rho", "h", "z_U", "q_U", "called", "inherited", "GRNBoost2_importance", "in_CollecTRI", "max_abs_z"]]
P.to_csv(OUT + "table7_pairs_in_all_three_regulons_ranked.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for n, t in (("1  pairs inside a regulon, by set", T1), ("2  within TFs that have a regulon", T2), ("3  top-N by |z| (Stage I pairs of TFs with a GRNBoost2 regulon)", T3), ("4  |z| in vs out of regulon", T4), ("5  |z| vs GRNBoost2 importance", T5)):
    print("\n== table", n); print(t.to_string(index=False))
print("\n== table 6 per TF (top 15 by pairs in GRNBoost2 regulon)"); print(g.head(15).to_string(index=False))
print("\n== table 7 top 25 pairs in all three regulons by max |z|", f"(total {len(P)})"); print(P.head(25).round(3).to_string(index=False))
