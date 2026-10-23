#!/usr/bin/env python3
"""How much of the covariation of each top-100 list (unshared = |z_U|, shared = |z_C|) is explained by cell-state axes?   (lists from build_top100_lists_alltf.py; clonal cells only)
Values: log2(CP10k/10 + 1) per gene.  Per pair (x, y) the covariation is measured at two levels
   within clones  : x and y centred on the clone mean, Pearson over all cells of clones with >= 2 cells        (the unshared component)
   between clones : Pearson of the clone means over clones with >= 2 cells                                      (the shared component)
Axes (cell scores):
   depth     = z(log total UMI)
   cycle     = Tirosh S and G2/M scores (mean z-scored log expression of the 43 / 54 genes in analysis_data/fm01/data/.../exclusion_list_B.json), both removed together
   identity  = PC1 of the BETWEEN-clone variation of 2,000 highly variable genes after regressing out depth and the two cycle scores (the main heritable axis); the cell score is its projection
For every pair, each axis is rebuilt WITHOUT the pair's own two genes (leave-one-out) so a gene is never regressed on itself.
Covariation removed = 1 - |rho after regressing the axes out| / |rho before|, at the same level (axes centred / averaged the same way); medians over the 100 pairs.
Output: analysis_data/fm_meta_programs_comparison/cell_state_axes/<dataset>_pairs.csv, <dataset>_summary.csv, <dataset>_identity_axis_genes.csv.    usage: list_cell_state_axes.py [FM01 FM06]"""
import os, sys, json
import numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; OUT = f"{A}/fm_meta_programs_comparison/cell_state_axes"; L = f"{A}/fm_meta_programs_comparison/top100_lists"; os.makedirs(OUT, exist_ok=True)
EX = json.load(open(f"{A}/fm01/data/twinscore_pdf_candidates_validation_20261014/exclusion_list_B.json"))
rng = np.random.default_rng(0)
def zs(M): M = M - M.mean(0); s = M.std(0); return M / np.where(s > 0, s, 1)
def resid(Y, D): return Y - D @ np.linalg.lstsq(D, Y, rcond=None)[0]
def corr(a, b): a = a - a.mean(); b = b - b.mean(); d = np.linalg.norm(a) * np.linalg.norm(b); return float(a @ b / d) if d > 0 else np.nan
def partial(x, y, Z):
    if Z is None or Z.shape[1] == 0: return corr(x, y)
    D = np.column_stack([np.ones(len(x)), Z - Z.mean(0)]); return corr(resid(x[:, None], D)[:, 0], resid(y[:, None], D)[:, 0])
for ds in (sys.argv[1:] or ["FM01", "FM06"]):
    qc = f"{ROOT}/finalized_data/{ds}_data/qc_filtered"; genes = np.array(open(f"{qc}/genes.txt").read().split()); cells = np.array(open(f"{qc}/cells.txt").read().split())
    X = sio.mmread(f"{qc}/{ds.lower()}_qc_counts.mtx").tocsr().astype(np.float32); obs = pd.read_csv(f"{qc}/obs_metadata.csv", index_col=0).loc[cells]
    cl = obs["fatemap_clone_singletcode"].astype(str).to_numpy(); ok = (cl != "nan") & (pd.Series(cl).str.len() > 0).to_numpy(); X = X[np.flatnonzero(ok)]; cl = cl[ok]
    n_c = pd.Series(cl).map(pd.Series(cl).value_counts()).to_numpy(); keep = n_c >= 2; print(f"\n{ds}: clonal cells {len(cl):,}; in clones >= 2 cells {int(keep.sum()):,} ({len(set(cl[keep])):,} clones)", flush=True)
    X = X[np.flatnonzero(keep)]; cl = cl[keep]; tot = np.asarray(X.sum(1)).ravel(); cp = X.multiply(1e4 / tot[:, None]).tocsr(); cp.data = np.log2(cp.data / 10 + 1); del X
    gi = {g: i for i, g in enumerate(genes)}; det = np.asarray((cp > 0).mean(0)).ravel(); mean = np.asarray(cp.mean(0)).ravel(); ex2 = np.asarray(cp.multiply(cp).mean(0)).ravel(); var = ex2 - mean ** 2
    cand = np.flatnonzero(det >= 0.05); disp = var[cand] / np.maximum(mean[cand], 1e-6); bins = pd.qcut(mean[cand], 20, labels=False, duplicates="drop"); dz = np.zeros(len(cand))
    for b in set(bins): m = bins == b; dz[m] = (disp[m] - disp[m].mean()) / (disp[m].std() + 1e-9)
    hv = cand[np.argsort(-dz)[:2000]]; Zh = zs(cp[:, hv].toarray()); hpos = {g: j for j, g in enumerate(genes[hv])}
    depth = zs(np.log(tot)[:, None])[:, 0]
    def setcols(names): idx = [gi[g] for g in names if g in gi and det[gi[g]] >= 0.05]; return idx
    Sg, Gg = setcols(EX["tirosh_S"]), setcols(EX["tirosh_G2M"]); ZS, ZG = zs(cp[:, Sg].toarray()), zs(cp[:, Gg].toarray()); sS, sG = ZS.sum(1), ZG.sum(1); posS = {g: j for j, g in enumerate(genes[Sg])}; posG = {g: j for j, g in enumerate(genes[Gg])}
    cyc = np.column_stack([sS / len(Sg), sG / len(Gg)])
    D0 = np.column_stack([np.ones(len(depth)), depth, cyc]); Rh = resid(Zh, D0)                                        # HVG expression without depth and cycle
    cid = pd.factorize(cl)[0]; ncl = cid.max() + 1; M = sp.csr_matrix((np.ones(len(cid)), (cid, np.arange(len(cid)))), shape=(ncl, len(cid))); cnt = np.asarray(M.sum(1)).ravel()
    cm = lambda V: (M @ V) / (cnt[:, None] if V.ndim > 1 else cnt)                                                    # clone means
    C = cm(Rh); C = C - C.mean(0); U, s, Vt = np.linalg.svd(C, full_matrices=False); v1 = Vt[0]; frac1 = float(s[0] ** 2 / (s ** 2).sum()); idn = Rh @ v1
    top = pd.DataFrame({"gene": genes[hv], "loading": v1}).sort_values("loading"); top.to_csv(f"{OUT}/{ds}_identity_axis_genes.csv", index=False)
    print(f" identity axis = PC1 of between-clone variation after removing depth and cycle: {frac1:.1%} of that variance. +end: {' '.join(top.gene.tail(12)[::-1])} | -end: {' '.join(top.gene.head(12))}")
    print(f" correlation among axes across cells: depth-S {np.corrcoef(depth, cyc[:,0])[0,1]:+.2f}, depth-G2M {np.corrcoef(depth, cyc[:,1])[0,1]:+.2f}, S-G2M {np.corrcoef(cyc[:,0], cyc[:,1])[0,1]:+.2f}, identity-depth {np.corrcoef(idn, depth)[0,1]:+.2f}, identity-S {np.corrcoef(idn, cyc[:,0])[0,1]:+.2f}, identity-G2M {np.corrcoef(idn, cyc[:,1])[0,1]:+.2f}")
    rows = []
    for lab in ("unshared", "shared"):
        P = pd.read_csv(f"{L}/{ds}_{lab}.csv")
        for r in P.itertuples():
            if r.TF not in gi or r.gene not in gi: continue
            x, y = cp[:, gi[r.TF]].toarray().ravel(), cp[:, gi[r.gene]].toarray().ravel(); drop = (r.TF, r.gene)
            def loo(sumv, pos, Z, n):
                s = sumv.copy(); k = n
                for g in drop:
                    if g in pos: s = s - Z[:, pos[g]]; k -= 1
                return s / max(k, 1)
            cS, cG = loo(sS, posS, ZS, len(Sg)), loo(sG, posG, ZG, len(Gg)); idn_p = idn.copy()
            for g in drop:
                if g in hpos: idn_p = idn_p - Rh[:, hpos[g]] * v1[hpos[g]]
            ax = {"depth": depth[:, None], "cycle": np.column_stack([cS, cG]), "identity": idn_p[:, None]}; ax["depth+cycle"] = np.column_stack([ax["depth"], ax["cycle"]]); ax["all three"] = np.column_stack([ax["depth"], ax["cycle"], ax["identity"]])
            for level in ("within", "between"):
                if level == "within": xs, ys, A_ = x - cm(x)[cid], y - cm(y)[cid], {k: v - cm(v)[cid] for k, v in ax.items()}
                else: xs, ys, A_ = cm(x), cm(y), {k: cm(v) for k, v in ax.items()}
                raw = corr(xs, ys); rec = dict(dataset=ds, list=lab, TF=r.TF, gene=r.gene, level=level, rho_raw=raw)
                for k, Z in A_.items(): rec["rho_" + k] = partial(xs, ys, Z)
                rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/{ds}_pairs.csv", index=False)
    S = []
    for (lab, level), d in R.groupby(["list", "level"], sort=False):
        rec = dict(dataset=ds, list=lab, level=level, n_pairs=len(d), median_abs_rho_raw=d.rho_raw.abs().median())
        for k in ("depth", "cycle", "identity", "depth+cycle", "all three"): rec["removed_" + k] = float(np.nanmedian(1 - d["rho_" + k].abs() / d.rho_raw.abs()))
        S.append(rec)
    S = pd.DataFrame(S); S.to_csv(f"{OUT}/{ds}_summary.csv", index=False); pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False), flush=True)
