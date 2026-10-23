"""PCA of the Novo Nordisk hPSC endoderm cells on log1p(CP10k) of the RAW RNA counts (instead of the authors' SCT-based Seurat PCA).
usage: pca_log1p_cp10k.py N_HVG SCOPE      SCOPE = all (global PCA over all 76,451 cells) or a SampleID (PCA over that sample's cells only; all of them, with or without a clone)
CP10k : per-cell totals over ALL 38,936 RNA genes (row sums of X, not the metadata nCount_RNA)
HVGs  : top N_HVG genes by Seurat-flavor normalized dispersion (scanpy flavor='seurat' logic: dispersion = var/mean of CP10k, log, 20 equal-width bins of log1p(mean),
        z-score within bin), computed over the cells of SCOPE
PCA   : log1p(CP10k) of the HVGs -> z-score per gene (clip +-10) -> 50 PCs (randomized SVD, seed 0)
output: export/pca_log1p_cp10k_hvg<N>_<SCOPE>.csv (cells x 50), ..._variance.csv, hvg_log1p_cp10k_<N>_<SCOPE>.csv
Reads the raw binary CSC export through memory maps.
"""
import sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

N_HVG = int(sys.argv[1]); SCOPE = sys.argv[2]
EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
nr, nall = [int(v) for v in open(f"{EXP}/RNA_counts_dim.txt").read().split()]
genes = pd.read_csv(f"{EXP}/RNA_counts_features.csv")["feature"].astype(str).values
allcells = pd.read_csv(f"{EXP}/cells.csv")["cell"].astype(str).values
if SCOPE == "all": cols = np.arange(nall)
else:
    obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False)
    assert (obs.index.astype(str).values == allcells).all(); cols = np.flatnonzero((obs["SampleID"].astype(str) == SCOPE).values); assert len(cols) > 0
nc = len(cols); cells = allcells[cols]
p = np.fromfile(f"{EXP}/RNA_counts_p.bin", dtype=np.float64).astype(np.int64)
I = np.memmap(f"{EXP}/RNA_counts_i.bin", dtype=np.int32, mode="r"); Xv = np.memmap(f"{EXP}/RNA_counts_x.bin", dtype=np.float64, mode="r")
st, ln = p[cols], p[cols + 1] - p[cols]
STEP = 2000
def chunks():
    for s in range(0, nc, STEP):
        e = min(nc, s + STEP); l = ln[s:e]; tot_n = int(l.sum())
        idx = np.repeat(st[s:e] - np.r_[0, np.cumsum(l)[:-1]], l) + np.arange(tot_n)          # global entry positions of these columns
        yield s, e, np.asarray(I[idx]), np.asarray(Xv[idx]), np.repeat(np.arange(e - s), l)

tot = np.zeros(nc)
for s, e, ii, xx, col in chunks(): tot[s:e] = np.bincount(col, weights=xx, minlength=e - s)
assert (tot > 0).all(); print(f"[{SCOPE}] pass 1 done: {nc} cells, median total {np.median(tot):.0f}", flush=True)

S1 = np.zeros(nr); S2 = np.zeros(nr)
for s, e, ii, xx, col in chunks():
    v = xx * (1e4 / tot[s + col]); S1 += np.bincount(ii, weights=v, minlength=nr); S2 += np.bincount(ii, weights=v * v, minlength=nr)
mean = S1 / nc; var = (S2 - nc * mean ** 2) / (nc - 1); mean[mean == 0] = 1e-12
disp = var / mean; disp[disp <= 0] = np.nan; disp = np.log(disp); lmean = np.log1p(mean)
df = pd.DataFrame({"disp": disp, "bin": pd.cut(lmean, bins=20)})
g = df.groupby("bin", observed=True)["disp"]; norm_disp = ((df["disp"] - g.transform("mean")) / g.transform("std")).to_numpy()
norm_disp[~np.isfinite(norm_disp)] = -np.inf
hv = np.sort(np.argsort(-norm_disp)[:N_HVG]); assert np.isfinite(norm_disp[hv]).all()
pd.DataFrame({"gene": genes[hv], "mean_cp10k": mean[hv], "norm_dispersion": norm_disp[hv]}).to_csv(f"{EXP}/hvg_log1p_cp10k_{N_HVG}_{SCOPE}.csv", index=False)
sct = set(pd.read_csv(f"{EXP}/SCT_variable_features.csv")["feature"].astype(str))
print(f"[{SCOPE}] pass 2 done: {N_HVG} HVGs, {len(sct & set(genes[hv]))} also in the authors' 3,000 SCT variable genes", flush=True)

gmap = np.full(nr, -1, np.int32); gmap[hv] = np.arange(len(hv))
D = np.zeros((nc, len(hv)), np.float32)
for s, e, ii, xx, col in chunks():
    gg = gmap[ii]; k = gg >= 0; D[s + col[k], gg[k]] = (xx[k] * (1e4 / tot[s + col[k]])).astype(np.float32)
np.log1p(D, out=D)
mu = D.mean(0); sdv = D.std(0, ddof=1); assert (sdv > 0).all()
D -= mu; D /= sdv; np.clip(D, -10, 10, out=D)
pca = PCA(n_components=50, svd_solver="randomized", random_state=0, copy=False); Z = pca.fit_transform(D)
pd.DataFrame(Z, index=pd.Index(cells, name="cell"), columns=[f"PC_{i}" for i in range(1, 51)]).to_csv(f"{EXP}/pca_log1p_cp10k_hvg{N_HVG}_{SCOPE}.csv")
pd.DataFrame({"PC": np.arange(1, 51), "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(f"{EXP}/pca_log1p_cp10k_hvg{N_HVG}_{SCOPE}_variance.csv", index=False)
print(f"[{SCOPE}] done: variance explained PC1-5 {np.round(pca.explained_variance_ratio_[:5], 3)}, 50 PCs {pca.explained_variance_ratio_.sum():.3f}", flush=True)
