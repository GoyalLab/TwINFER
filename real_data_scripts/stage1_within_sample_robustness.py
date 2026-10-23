#!/usr/bin/env python3
"""Within-sample robustness of the all-TF Stage I results.
Inputs (analysis_data/<fm>/data/stage1_tf_ge20/): pooled pdf_procedure_all_candidate_pairs.csv (or merged part files), and sample_<S>/pdf_procedure_all_candidate_pairs.csv for each replicate S (Stage I run on that replicate's cells only).
Reports: Stage I counts, pairs that clear Stage I (q <= 0.05) in the pooled run AND in every replicate with the same sign of z_rho, z_rho correlations, top-N replication, and the share of pairs with z_C > 2.326
by TF h_x quartile pooled vs within each replicate. Memory-light: integer pair keys, float32.   usage: stage1_within_sample_robustness.py FM06 A B      (replicate labels as in obs_metadata.csv)"""
import sys, numpy as np, pandas as pd
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; ds = sys.argv[1]; reps = sys.argv[2:]; D = f"{ROOT}/analysis_data/{ds.lower()}/data/stage1_tf_ge20"
cols = ["TF", "gene", "z_rho", "stage1", "h_x", "h", "inherited", "called"]; dt = {"TF": "category", "gene": "category", "z_rho": "float32", "stage1": "bool", "h_x": "float32", "h": "float32", "inherited": "bool", "called": "bool"}
tfs = pd.read_csv(f"{D}/tf_list.csv").TF.tolist(); tcode = {t: i for i, t in enumerate(tfs)}; gcode = {}
def load(f):
    X = pd.read_csv(f, usecols=cols, dtype=dt)
    for g in X.gene.cat.categories: gcode.setdefault(g, len(gcode))
    k = X.TF.astype(str).map(tcode).to_numpy(np.int64) * 100000 + X.gene.astype(str).map(gcode).to_numpy(np.int64)
    X["key"] = k; X = X.drop(columns=["TF", "gene"]).set_index("key"); return X
import glob, os
pf = f"{D}/pdf_procedure_all_candidate_pairs.csv"
if os.path.exists(pf): P = load(pf)
else:   # TF subsets (FM01): each subset was Benjamini-Hochberg corrected on its own, so Stage I is recomputed over the merged set from z_rho
    from scipy.stats import norm
    P = pd.concat([load(f) for f in sorted(glob.glob(f"{D}/part*/pdf_procedure_all_candidate_pairs.csv"))])
    p_ = 2 * norm.sf(np.abs(P.z_rho.to_numpy(np.float64))); o = np.argsort(p_); qq_ = p_[o] * len(p_) / (np.arange(len(p_)) + 1); qq_ = np.minimum.accumulate(qq_[::-1])[::-1]; q_ = np.empty_like(qq_); q_[o] = np.minimum(qq_, 1)
    P["stage1"] = q_ <= 0.05; P["called"] = False   # 'called' (q_U) is not recomputed for merged subsets
    print("merged TF subsets; Stage I recomputed over all", len(P), "pairs")
S = {r: load(f"{D}/sample_{r}/pdf_procedure_all_candidate_pairs.csv") for r in reps}
idx = P.index
for r in reps: idx = idx.intersection(S[r].index)
print(f"{ds}: pooled pairs {len(P):,}; " + "; ".join(f"{r} {len(S[r]):,}" for r in reps) + f"; in all {len(idx):,}")
P = P.loc[idx]; S = {r: S[r].loc[idx] for r in reps}
hx_tf = P.h_x.groupby(np.asarray(idx) // 100000).first()
same = np.ones(len(idx), bool); ok = P.stage1.to_numpy().copy()
for r in reps: same &= (np.sign(P.z_rho.to_numpy()) == np.sign(S[r].z_rho.to_numpy())); ok &= S[r].stage1.to_numpy()
keep = ok & same
print("Stage I (q<=0.05): pooled {:,}".format(int(P.stage1.sum())) + "".join(f" | {r} {int(S[r].stage1.sum()):,}" for r in reps))
print(f"kept (pooled & every replicate, same sign): {int(keep.sum()):,} = {100 * keep.sum() / P.stage1.sum():.1f}% of pooled Stage I; sign flips among pooled Stage I: {int((P.stage1.to_numpy() & ~same).sum()):,}")
z = pd.DataFrame({"P": P.z_rho.to_numpy(), **{r: S[r].z_rho.to_numpy() for r in reps}}); print("z_rho correlations:", {f"{a}-{b}": round(float(z[a].corr(z[b])), 3) for i, a in enumerate(z.columns) for b in z.columns[i + 1:]})
az = np.abs(z.P.to_numpy())
for n in (100, 1000, 10000, 100000):
    top = np.argsort(-az)[:n]; print(f"  top {n:>6,} pairs by |z_rho| pooled: pass in every replicate with the same sign: {100 * keep[top].mean():.1f}%")
q = pd.qcut(hx_tf, 4, labels=["Q1 low h_x", "Q2", "Q3", "Q4 high h_x"]); qq = pd.Series(np.asarray(idx) // 100000).map(q).to_numpy()
out = pd.DataFrame({"pooled": pd.Series(P.inherited.to_numpy()).groupby(qq).mean() * 100, **{f"within {r}": pd.Series(S[r].inherited.to_numpy()).groupby(qq).mean() * 100 for r in reps}}).round(1)
print("share of pairs with z_C > 2.326 by TF h_x quartile:\n", out)
print("median z_C: pooled %.2f" % P.h.median() + "".join(f", {r} {S[r].h.median():.2f}" for r in reps))
print("kept pairs: inherited pooled %.1f%%" % (100 * P.inherited.to_numpy()[keep].mean()) + "".join(f", {r} {100 * S[r].inherited.to_numpy()[keep].mean():.1f}%" for r in reps) + "; called pooled %.1f%%" % (100 * P.called.to_numpy()[keep].mean()))
print("TF-level sister correlation h_x median: pooled %.3f" % hx_tf.median() + "".join(f", {r} {S[r].h_x.groupby(np.asarray(idx) // 100000).first().median():.3f}" for r in reps))
inv = {v: k for k, v in gcode.items()}; kk = np.asarray(idx)[keep]
pd.DataFrame({"TF": [tfs[i] for i in kk // 100000], "gene": [inv[i] for i in kk % 100000], "z_rho_pooled": z.P.to_numpy()[keep], **{f"z_rho_{r}": z[r].to_numpy()[keep] for r in reps}}).to_csv(f"{D}/stage1_pooled_and_all_samples_zrho.csv.gz", index=False)
