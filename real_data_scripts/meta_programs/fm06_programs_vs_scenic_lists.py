#!/usr/bin/env python3
"""FM06: how the 6 robust pooled NMF programs (analysis_data/fm06/data/mp_nmf_pooled/robust_programs.csv, top 50 genes) relate to the SCENIC regulons (analysis_data/fm06/pyscenic: regulons_FM06.csv, auc_FM06.csv)
and to the shared / unshared top lists.  (Not the Gavish meta-programs.)  Must run with envs/pyscenic/bin/python (loompy) and LD_LIBRARY_PATH=/home/gzu5140/.conda/envs/grnboost310/lib.
 A. gene overlap: program genes vs regulon target set (+ the TF itself), hypergeometric over the 9,100 expressed genes
 B. activity: program score (mean z-scored log1p(CP10k) of its genes) vs regulon AUCell, across cells and at two levels
      between clones = correlation of clone means (clones >= 2 cells)  [the shared component]      within clones = clone-centred cells  [the unshared component]
    and the same with the genes shared by program and regulon removed from both sides (regulon score = mean z of its targets, not AUCell) to remove the circularity
 C. lists: for each program, the pairs of the top-1000 shared / unshared lists (ChIP-Atlas edges, Stage I, no gene filter; AEBP1, CENPA, HMGA1, YBX1 and non-Lambert regulators excluded) whose TARGET is in the program,
    the fold over the share of all Stage I pairs, and how many of these pairs lie inside a GRNBoost2 SCENIC regulon.
Output: analysis_data/fm06/pyscenic/compare/programs_vs_scenic_lists/*.csv"""
import ast, os, sys
import numpy as np, pandas as pd, loompy
from scipy.stats import hypergeom
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; S = f"{A}/fm06/pyscenic"; OUT = f"{S}/compare/programs_vs_scenic_lists"; os.makedirs(OUT, exist_ok=True)
pr = pd.read_csv(f"{A}/fm06/data/mp_nmf_pooled/robust_programs.csv"); P = {f"K{r.K}_{r.idx}": r.genes.split() for r in pr.itertuples()}
with loompy.connect(f"{S}/expr_FM06.loom") as ds: genes = ds.ra["Gene"].astype(str); cells = ds.ca["CellID"].astype(str); X = ds[:, :].astype(np.float32)
gi = {g: i for i, g in enumerate(genes)}; Z = (X - X.mean(1, keepdims=True)) / np.where(X.std(1, keepdims=True) > 0, X.std(1, keepdims=True), 1); del X
auc = pd.read_csv(f"{S}/auc_FM06.csv", index_col=0).loc[cells]; regs = [c[:-3] for c in auc.columns]
rr = pd.read_csv(f"{S}/regulons_FM06.csv", skiprows=3, header=None, names=["TF", "MotifID", "AUC", "NES", "MSQ", "OI", "Annotation", "Context", "TargetGenes", "RankAtMax"])
RT = {}
for r in rr.itertuples():
    if r.TF in regs: RT.setdefault(r.TF, set()).update(g for g, w in ast.literal_eval(r.TargetGenes))
clones = pd.read_csv(f"{S}/clones_FM06.csv").set_index("CellID").loc[cells].clone.astype(str).to_numpy(); n_c = pd.Series(clones).map(pd.Series(clones).value_counts()).to_numpy(); k2 = n_c >= 2
cid = pd.factorize(clones[k2])[0]; cnt = np.bincount(cid); cm = lambda v: np.bincount(cid, weights=v) / cnt
def score(gs, drop=()): j = [gi[g] for g in gs if g in gi and g not in drop]; return Z[j].mean(0) if j else None
def levels(a, b):
    a2, b2 = a[k2], b[k2]; return np.corrcoef(a, b)[0, 1], np.corrcoef(cm(a2), cm(b2))[0, 1], np.corrcoef(a2 - cm(a2)[cid], b2 - cm(b2)[cid])[0, 1]
LAB = {}
for p, g in P.items(): LAB[p] = " ".join(g[:6])
print("programs:", LAB, flush=True); N = len(genes)
ov, cr = [], []
for p, g in P.items():
    sp_ = score(g); gs = set(g) & set(genes)
    for t in regs:
        T = RT[t] | {t}; k = len(gs & T); ov.append(dict(program=p, regulon=t, n_targets=len(RT[t]), overlap=k, p=hypergeom.sf(k - 1, N, len(T & set(genes)), len(gs)) if k else 1.0))
        a_ = auc[t + "(+)"].to_numpy(); c_all, c_bt, c_wi = levels(sp_, a_)
        sh = gs & T; sp2 = score(g, drop=sh); rs2 = score(sorted(RT[t] - set(g)))
        if sp2 is not None and rs2 is not None and len(sh) > 0: i_all, i_bt, i_wi = levels(sp2, rs2)
        else: i_all = i_bt = i_wi = np.nan
        cr.append(dict(program=p, regulon=t, corr_cells=c_all, corr_between=c_bt, corr_within=c_wi, overlap_genes=len(sh), corr_between_noOverlap=i_bt, corr_within_noOverlap=i_wi))
OV = pd.DataFrame(ov); CR = pd.DataFrame(cr); OV.to_csv(f"{OUT}/program_regulon_overlap.csv", index=False); CR.to_csv(f"{OUT}/program_regulon_activity_correlation.csv", index=False)
pd.set_option("display.width", 250)
print("\nA/B. per program: top regulons by gene overlap, by between-clone correlation (shared level) and by within-clone correlation (unshared level); in brackets the correlation after removing shared genes")
for p in P:
    o = OV[OV.program == p].sort_values("overlap", ascending=False).head(4); c = CR[CR.program == p]
    b = c.reindex(c.corr_between.abs().sort_values(ascending=False).index).head(4); w = c.reindex(c.corr_within.abs().sort_values(ascending=False).index).head(4)
    print(f"\n{p} ({LAB[p]})\n  overlap : " + "; ".join(f"{r.regulon} {r.overlap}/50 (p={r.p:.0e})" for r in o.itertuples()) + "\n  between : " + "; ".join(f"{r.regulon} {r.corr_between:+.2f} [{r.corr_between_noOverlap:+.2f}]" for r in b.itertuples()) + "\n  within  : " + "; ".join(f"{r.regulon} {r.corr_within:+.2f} [{r.corr_within_noOverlap:+.2f}]" for r in w.itertuples()))
KEY = [t for t in ["MITF", "SOX10", "TFAP2A", "TFAP2C", "FOSL1", "FOSL2", "JUN", "JUNB", "FOS", "E2F1", "FOXM1", "MYBL2", "MYC", "SNAI2", "SMAD3", "ETV1"] if t in regs]
print("\nkey regulons present:", KEY); print("correlation program score vs regulon AUCell: between clones / within clones")
print(pd.DataFrame({t: {p: f"{CR[(CR.program == p) & (CR.regulon == t)].corr_between.iloc[0]:+.2f}/{CR[(CR.program == p) & (CR.regulon == t)].corr_within.iloc[0]:+.2f}" for p in P} for t in KEY}).to_string())
# C. lists
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
F = pd.read_csv(f"{S}/compare/twinscore_pairs_vs_regulons/fm06_twinscore_pairs_with_regulon_flags.csv.gz", usecols=["TF", "gene", "h", "z_U", "stage1", "in_GRNBoost2_regulon", "any_regulon"]); F = F[F.stage1]; F = F[~(F.TF.isin(four) | ~F.TF.isin(lam))].reset_index(drop=True)
rows = []; N1000 = 1000
lists = {"unshared": F.assign(s=F.z_U.abs()).nlargest(N1000, "s"), "shared": F.assign(s=F.h.abs()).nlargest(N1000, "s")}
for p, g in P.items():
    gset = set(g); base = F.gene.isin(gset).mean()
    for lab, d in lists.items():
        m = d.gene.isin(gset); rows.append(dict(program=p, list=lab, pairs_target_in_program=int(m.sum()), pct=100 * m.mean(), baseline_pct=100 * base, fold=m.mean() / base, pct_of_those_in_GRNBoost2_regulon=100 * d[m].in_GRNBoost2_regulon.mean() if m.any() else np.nan, top_TFs=", ".join(f"{k}:{v}" for k, v in d[m].TF.value_counts().head(3).items())))
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/program_vs_lists_top1000.csv", index=False)
print(f"\nC. top-1000 lists (ChIP-Atlas edges): pairs whose target is in the program (n, % of 1000, fold over all Stage I pairs), % of those inside a SCENIC regulon, main TFs")
print(C.assign(pct=lambda x: x.pct.round(1), fold=lambda x: x.fold.round(1), pct_of_those_in_GRNBoost2_regulon=lambda x: x.pct_of_those_in_GRNBoost2_regulon.round(0)).drop(columns="baseline_pct").to_string(index=False))
print("\nreference: share of all Stage I pairs inside a GRNBoost2 regulon %.1f%%; of the unshared top-1000 %.1f%%; of the shared top-1000 %.1f%%" % (100 * F.in_GRNBoost2_regulon.mean(), 100 * lists["unshared"].in_GRNBoost2_regulon.mean(), 100 * lists["shared"].in_GRNBoost2_regulon.mean()))
