#!/usr/bin/env python3
"""The three ranked lists of the all-TF Stage I run: listed pairs (Stage I, q_rho <= 0.05, after the PDF's gene filter) ranked by |z_rho|, by |z_C| (column h = sign(S) C / sigma_C) and by |z_U|.
Per dataset: analysis_data/<fm>/data/stage1_tf_ge20/top_lists/top{N}_by_{z_rho,z_C,z_U}.csv with rho, S, C, U, h_x, h_y, the three z-scores, q_U, called/inherited flags, the rank of the pair in each of the three lists, and
robust = the pair also clears Stage I in every replicate with the same sign (from stage1_within_sample_robustness.py; blank when that check has not been run).   usage: stage1_top_lists.py FM06 [N=1000]
Non-classic regulators removed before ranking (user request): AEBP1, CENPA, HMGA1, YBX1 (DROP_TF); lists go to top_lists_classic_TFs/. The Lambert assessment, DBD, binding mode and motif status of each TF are added as columns (analysis_data/lambert_tf_assessment.csv).
Stage I membership (q_rho) was computed over all pairs including these TFs and is not recomputed."""
import os, sys, numpy as np, pandas as pd
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; ds = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; D = f"{ROOT}/analysis_data/{ds.lower()}/data/stage1_tf_ge20"
cols = ["TF", "gene", "rho", "S", "C", "U", "h_x", "h_y", "z_rho", "h", "z_U", "q_rho", "q_U", "stage1", "called", "inherited", "filtered_out"]
dt = {c: "float32" for c in ["rho", "S", "C", "U", "h_x", "h_y", "z_rho", "h", "z_U", "q_rho", "q_U"]}; dt.update(TF="category", gene="category", stage1="bool", called="bool", inherited="bool", filtered_out="bool")
f = f"{D}/pdf_procedure_all_candidate_pairs.csv"
if os.path.exists(f): A = pd.read_csv(f, usecols=cols, dtype=dt)
else:
    import glob; from scipy.stats import norm
    A = pd.concat([pd.read_csv(g, usecols=cols, dtype=dt) for g in sorted(glob.glob(f"{D}/part*/pdf_procedure_all_candidate_pairs.csv"))], ignore_index=True)
    p = 2 * norm.sf(np.abs(A.z_rho.to_numpy(np.float64))); o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1)
    A["q_rho"] = qq; A["stage1"] = qq <= 0.05
    s1 = A.stage1.to_numpy(); pu = 2 * norm.sf(np.abs(A.z_U.to_numpy(np.float64)[s1])); o = np.argsort(pu); q = pu[o] * len(pu) / (np.arange(len(pu)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; qu = np.empty_like(q); qu[o] = np.minimum(q, 1)
    A["q_U"] = np.nan; A.loc[s1, "q_U"] = qu; A["called"] = A.q_U <= 0.05; A["inherited"] = A.h > 2.326   # merged TF subsets: BH recomputed over all pairs
DROP_TF = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
L = A[A.stage1 & ~A.filtered_out & ~A.TF.astype(str).isin(DROP_TF)].drop(columns=["stage1", "filtered_out"]).copy(); L.rename(columns={"h": "z_C"}, inplace=True)
for z in ("z_rho", "z_C", "z_U"): L["rank_" + z] = L[z].abs().rank(ascending=False, method="first").astype(int)
rob = f"{D}/stage1_pooled_and_all_samples_zrho.csv.gz"
if os.path.exists(rob):
    R = pd.read_csv(rob, usecols=["TF", "gene"]); R["robust"] = True; L = L.merge(R, on=["TF", "gene"], how="left"); L["robust"] = L.robust.fillna(False)
else: L["robust"] = np.nan
LA = pd.read_csv(f"{ROOT}/analysis_data/lambert_tf_assessment.csv"); L["TF"] = L.TF.astype(str); L = L.merge(LA, on="TF", how="left")
out = f"{D}/top_lists_classic_TFs"; os.makedirs(out, exist_ok=True)
print(f"{ds}: pairs {len(A):,}; Stage I {int(A.stage1.sum()):,}; listed (after gene filter, without {sorted(DROP_TF)}) {len(L):,}; robust-flag available: {os.path.exists(rob)}")
for z in ("z_rho", "z_C", "z_U"):
    T = L.reindex(L[z].abs().sort_values(ascending=False).index).head(N).reset_index(drop=True); T.insert(0, "rank", np.arange(1, len(T) + 1)); T.to_csv(f"{out}/top{N}_by_{z}.csv", index=False)
    r = T.head(100); print(f"  by |{z}|: top-100 median |z_rho| {r.z_rho.abs().median():.1f} |z_C| {r.z_C.abs().median():.1f} |z_U| {r.z_U.abs().median():.1f}; called {100 * r.called.mean():.0f}% inherited {100 * r.inherited.mean():.0f}%; robust {100 * r.robust.astype(float).mean():.0f}%; TFs in top 100: {r.TF.nunique()}; most common TFs: " + ", ".join(f"{k}({v})" for k, v in r.TF.astype(str).value_counts().head(4).items()))
    print("    top 5:", "; ".join(f"{a}-{b} (rho {x:.2f}, zC {c:.1f}, zU {u:.1f})" for a, b, x, c, u in zip(T.TF[:5], T.gene[:5], T.rho[:5], T.z_C[:5], T.z_U[:5])))
