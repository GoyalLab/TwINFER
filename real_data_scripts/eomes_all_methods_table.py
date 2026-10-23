#!/usr/bin/env python3
"""EOMES targets across all methods (TwinScore weighted rho / S / U / z_U, Pearson, |Spearman|, GRNBoost2, GENIE3, partial correlations) on yscher's endo T15 cells.
Universe = the PDF's listed pairs for EOMES (Stage I pass, gene filter applied); every rank is among those pairs, as in the PDF's tables.
usage: eomes_all_methods_table.py [TF=EOMES]"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
TF = sys.argv[1] if len(sys.argv) > 1 else "EOMES"
B = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_20260927/data"
T = f"{B}/twinscore_pdf_candidates_yscher_endo"; C = f"{B}/competitors_yscher_endo"; X = f"{B}/yscher_endo_T15/expr_genes_x_cells.csv"
A = pd.read_csv(f"{T}/pdf_procedure_all_candidate_pairs.csv"); E = A[(A.TF == TF) & A.stage1 & ~A.filtered_out].copy()
# competitor scores: re-assemble from the npz files so a late GENIE3 chunk is picked up (assemble step may have run before it finished)
sc = pd.read_csv(f"{C}/competitor_scores_candidate_pairs.csv").rename(columns={"TF": "TF", "gene": "gene", "rho": "spearman_abs"})
E = E.merge(sc[sc.TF == TF], on=["TF", "gene"], how="left")
if "genie3" not in E.columns or E.genie3.isna().all():
    import glob
    fs = sorted(glob.glob(f"{C}/genie3_chunk*.npz"))
    if fs:
        z0 = np.load(fs[0], allow_pickle=True); tfs = list(z0["tfs"]); genes = list(z0["genes"]); gi = {g: i for i, g in enumerate(genes)}; ti = tfs.index(TF)
        M = np.full(len(genes), np.nan, np.float32)
        for f in fs:
            z = np.load(f, allow_pickle=True); M[z["targets"]] = z["M"][ti]
        E["genie3"] = E.gene.map(lambda g: M[gi[g]] if g in gi else np.nan)
need = set(E.gene) | {TF}; rows = {}
for ch in pd.read_csv(X, index_col=0, chunksize=2000):
    for g in ch.index.intersection(list(need)): rows[g] = ch.loc[g].to_numpy(float)
E = E[E.gene.isin(rows)].copy(); E["pearson"] = [np.corrcoef(rows[TF], rows[g])[0, 1] for g in E.gene]
meas = {"weighted rho": "rho", "S": "S", "U": "U", "Pearson": "pearson", "|Spearman|": "spearman_abs", "GRNBoost2": "grnboost2", "GENIE3": "genie3", "ppcor": "ppcor", "ppcor_shrunk": "ppcor_shrunk"}
meas = {k: v for k, v in meas.items() if v in E.columns and E[v].notna().any()}
E = E.dropna(subset=list(meas.values())).copy().reset_index(drop=True)
for k, c in meas.items(): E["rank_" + k] = E[c].abs().rank(ascending=False, method="first").astype(int)
E["rank_TwinScore |z_U|"] = E.z_U.abs().rank(ascending=False, method="first").astype(int)
E["z_s"] = E.h                                                    # shared-part z = h = sign(S) C / sigma_C (PDF Stage II; handoff z_C)
E["rank_TwinScore z_s"] = E.z_s.rank(ascending=False, method="first").astype(int)   # high = shared (inherited), as yscher ranks z_C
E.sort_values("rank_weighted rho").to_csv(f"{T}/{TF}_all_methods_listed_pairs.csv", index=False)
names = list(meas) + ["TwinScore |z_U|", "TwinScore z_s"]
print(f"{TF}: {len(E)} listed targets; methods present: {list(meas)}")
print("\nrank correlation among methods (Spearman of the rankings by |score|):")
R = pd.DataFrame([[spearmanr(E["rank_" + a], E["rank_" + b])[0] for b in names] for a in names], index=names, columns=names); print(R.round(2).to_string())
print("\ntop-50 overlap between methods:")
top = {n: set(E.gene[E["rank_" + n] <= 50]) for n in names}
print(pd.DataFrame([[len(top[a] & top[b]) for b in names] for a in names], index=names, columns=names).to_string())
print("\ntop 12 per method: gene (rank in TwinScore|z_U| / z_s / weighted rho / GRNBoost2 / GENIE3 / ppcor)")
for n in names:
    t = E.sort_values("rank_" + n).head(12); print(f"--- {n}: " + ", ".join(f"{g}({a}/{z}/{b}/{c}/{d}/{e})" for g, a, z, b, c, d, e in zip(t.gene, t["rank_TwinScore |z_U|"], t["rank_TwinScore z_s"], t["rank_weighted rho"], t.get("rank_GRNBoost2", t.index), t.get("rank_GENIE3", t.index * 0), t.get("rank_ppcor", t.index))))
