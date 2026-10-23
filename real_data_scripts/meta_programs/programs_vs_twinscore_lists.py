#!/usr/bin/env python3
"""Connect the NMF programs (twin similarity of their genes) to the TwinScore TF-target lists (z_rho, h = z_C, z_U).
For every dataset with a TwinScore all-candidate-pairs table (FM01, FM03, FM05, FM06, FM07) and every highlighted program group (twin_vs_random_logfc.build_groups, top-50 genes):
  pairs whose TARGET is a program gene (Stage I pairs, no gene filter, as in the PDF top-100 lists)
  A. how many Stage I pairs, and enrichment of those pairs in the top-N lists by |z_rho|, |h|, |z_U| (fold over the program's share of all Stage I pairs; hypergeometric p)
  B. median |z_rho|, |h|, |z_U| of those pairs against all Stage I pairs, and the share-of-S that is shared, C/S (pairs with S > 0.05)
  C. the program's twin reduction (program_level_summary.csv: reduction of |log2 FC| in twins vs random pairs; excess over matched controls) -> Spearman across programs with the median z of its pairs
Outputs: analysis_data/fm_meta_programs_comparison/programs_vs_twinscore_lists/{per_program.csv, topN_enrichment.csv}
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import hypergeom, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_vs_random_logfc as W
ROOT = W.ROOT; SRC = os.environ.get("PROG_SRC", "mp_nmf"); SUF = "_pooled" if SRC == "mp_nmf_pooled" else ""; OUT = f"{ROOT}/analysis_data/fm_meta_programs_comparison/programs_vs_twinscore_lists"; os.makedirs(OUT, exist_ok=True)
rows = []; enr = []
for ds in ["FM01", "FM03", "FM05", "FM06", "FM07"]:
    A = pd.read_csv(f"{ROOT}/analysis_data/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv"); A = A[A.stage1].copy()
    for z in ("z_rho", "h", "z_U"): A["a" + z] = A[z].abs()
    A["CS"] = np.where(A.S > 0.05, A.C / A.S, np.nan); N = len(A)
    G = W.build_groups(ds, 50, SRC); PL = pd.read_csv(f"{ROOT}/analysis_data/{ds.lower()}/data/twin_vs_random_logfc_top50/program_level_summary{SUF}.csv").set_index("group")
    for n, genes in G.items():
        ing = A.gene.isin(genes); k = int(ing.sum()); d = A[ing]
        r = dict(dataset=ds, group=n, n_program_genes=len(genes), stage1_pairs_total=N, pairs_target_in_program=k, share_of_stage1_pct=round(100 * k / N, 2),
                 median_abs_z_rho=d.az_rho.median(), median_abs_z_rho_all=A.az_rho.median(), median_abs_h=d.ah.median(), median_abs_h_all=A.ah.median(), median_abs_z_U=d.az_U.median(), median_abs_z_U_all=A.az_U.median(),
                 median_C_over_S=d.CS.median(), median_C_over_S_all=A.CS.median(), pct_inherited=100 * d.inherited.mean(), pct_inherited_all=100 * A.inherited.mean(), pct_called=100 * d.called.mean(), pct_called_all=100 * A.called.mean())
        if n in PL.index: r.update(prog_reduction=PL.loc[n, "rel_reduction_program"], ctrl_reduction=PL.loc[n, "rel_reduction_matched_controls"], excess_reduction=PL.loc[n, "rel_reduction_program"] - PL.loc[n, "rel_reduction_matched_controls"])
        rows.append(r)
        for z in ("az_rho", "ah", "az_U"):
            for topn in (100, 1000):
                top = A.nlargest(topn, z); x = int(top.gene.isin(genes).sum()); p = hypergeom.sf(x - 1, N, k, topn)
                enr.append(dict(dataset=ds, group=n, ranked_by="|" + z[1:] + "|", top_N=topn, pairs_in_program=x, expected=round(topn * k / N, 1), fold=round(x / max(topn * k / N, 1e-9), 2), p=p))
R = pd.DataFrame(rows); E = pd.DataFrame(enr); R.to_csv(f"{OUT}/per_program{SUF}.csv", index=False); E.to_csv(f"{OUT}/topN_enrichment{SUF}.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
c = ["dataset", "group", "pairs_target_in_program", "share_of_stage1_pct", "median_abs_z_rho", "median_abs_z_rho_all", "median_abs_h", "median_abs_h_all", "median_abs_z_U", "median_abs_z_U_all", "median_C_over_S", "median_C_over_S_all", "pct_inherited", "pct_inherited_all", "prog_reduction", "excess_reduction"]
print(R[c].round(2).to_string(index=False))
print("\nSpearman across programs (n=%d):" % R.prog_reduction.notna().sum())
for y in ("median_abs_z_rho", "median_abs_h", "median_abs_z_U", "median_C_over_S", "pct_inherited", "pct_called"):
    for x in ("prog_reduction", "excess_reduction"):
        m = R[[x, y]].dropna(); print(f"  {x:17s} vs {y:18s}: rho = {spearmanr(m[x], m[y])[0]:+.2f}")
print("\ntop-1000 enrichment, folds over expectation:"); P = E[E.top_N == 1000].pivot_table(index=["dataset", "group"], columns="ranked_by", values="fold"); print(P.round(2).to_string())
