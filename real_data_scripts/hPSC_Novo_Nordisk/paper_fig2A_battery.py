"""Fig 2A battery (hESCs p2 = hESCs_p2_parent; paper: between ~0.55, within ~0.68, Cohen's d = 3.51).
Every measure is oriented so that HIGHER = MORE SIMILAR (distances are negated); Cohen's d = (mean within - mean between) / pooled SD over 17,497 same-clone pairs vs 300,000 random different-clone pairs.
Groups: A expression correlation on gene subsets (SCT data, log), B rank / cosine variants, C scale.data (Pearson residuals), D embeddings (PCs at several depths, UMAP rna / rpca as distance),
E nearest-neighbour graph (SCT SNN / NN weights), F neighbour-smoothed expression. Output: CSV + dot plot in analysis_data/hPSC_Novo_Nordisk/paper_reproduction/."""
import json, numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.stats import mannwhitneyu, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"
H5 = "/home/gzu5140/TwINFER_KA/real_data/Novo_nordisk_hPSC_endo.h5ad"; EXCL = "/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/gene_exclusion_list.json"
rng = np.random.default_rng(0); obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False)

def load_dense(prefix, cols, nr):
    p = np.fromfile(f"{EXP}/{prefix}_p.bin", dtype=np.float64).astype(np.int64); I = np.memmap(f"{EXP}/{prefix}_i.bin", dtype=np.int32, mode="r"); X = np.memmap(f"{EXP}/{prefix}_x.bin", dtype=np.float64, mode="r")
    D = np.zeros((len(cols), nr), np.float32)
    for s in range(0, len(cols), 2000):
        c = cols[s:s + 2000]; l = p[c + 1] - p[c]; idx = np.repeat(p[c] - np.r_[0, np.cumsum(l)[:-1]], l) + np.arange(int(l.sum())); D[s + np.repeat(np.arange(len(c)), l), np.asarray(I[idx])] = np.asarray(X[idx])
    return D
def pearson_rows(D): D -= D.mean(1, keepdims=True); D /= np.maximum(np.sqrt(np.einsum("ij,ij->i", D, D) / D.shape[1]), 1e-9)[:, None]; return (D @ D.T) / D.shape[1]
def cosine_rows(D): D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9); return D @ D.T
def spearman_rows(D): return pearson_rows(np.apply_along_axis(rankdata, 1, D).astype(np.float32))
def cohen_d(a, b): return (a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
def pairs_within(l):
    o = np.argsort(l, kind="stable"); ls = l[o]; st = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]]); en = np.r_[st[1:], len(ls)]; A, B = [], []
    for s, e in zip(st, en):
        if e - s >= 2: i, j = np.triu_indices(e - s, 1); A.append(o[s + i]); B.append(o[s + j])
    return np.concatenate(A), np.concatenate(B)
def pdist_sim(Z, a, b): return -np.linalg.norm(Z[a] - Z[b], axis=1)

m = ((obs.SampleID.astype(str) == "hESCs_p2_parent") & obs["clone.id"].notna()).to_numpy(); cols = np.flatnonzero(m); clone = obs["clone.id"][m].astype(int).to_numpy()
a, b = pairs_within(clone); ra = rng.integers(0, len(cols), 600_000); rb = rng.integers(0, len(cols), 600_000); ok = clone[ra] != clone[rb]; ra, rb = ra[ok][:300_000], rb[ok][:300_000]
rows = []
def rep(group, name, w, bt):
    w = np.asarray(w, np.float64); bt = np.asarray(bt, np.float64)
    rows.append(dict(group=group, measure=name, mean_between=bt.mean(), mean_within=w.mean(), shift=w.mean() - bt.mean(), cohens_d=cohen_d(w, bt), AUC=mannwhitneyu(w, bt, alternative="greater").statistic / (len(w) * len(bt))))
    print(rows[-1], flush=True)
def rep_mat(group, name, R): rep(group, name, R[a, b], R[ra, rb])

sct = pd.read_csv(f"{EXP}/SCT_counts_features.csv").feature.astype(str).values; var = pd.read_csv(f"{EXP}/SCT_variable_features.csv").feature.astype(str).values
D = load_dense("SCT_data", cols, len(sct)); ex = json.load(open(EXCL)); gi = {g: i for i, g in enumerate(sct)}
# mean = D.mean(0); varg = D.var(0); order_mean = np.argsort(-mean); order_var = np.argsort(-varg)      # replaced: D.var(0) makes full-size temporaries
mean = D.mean(0); varg = np.empty(D.shape[1]); frac_det = np.empty(D.shape[1])
for s0 in range(0, D.shape[1], 2000):
    blk = D[:, s0:s0 + 2000]; varg[s0:s0 + 2000] = blk.var(0); frac_det[s0:s0 + 2000] = (blk > 0).mean(0)
order_mean = np.argsort(-mean); order_var = np.argsort(-varg)
def idx_of(names): return np.array([gi[g] for g in names if g in gi])
# ---- A: Pearson on gene subsets of SCT data (log)
subsets = {"3,000 SCT variable genes": idx_of(var)}      # "all genes" is done last, in place (a copy does not fit in memory)
for n in (200, 500, 1000, 2000, 5000, 10000): subsets[f"top {n} by mean expression"] = order_mean[:n]
for n in (500, 1000, 2000, 5000): subsets[f"top {n} by variance"] = order_var[:n]
subsets["cell-cycle genes (Tirosh S + G2M)"] = idx_of(ex["tirosh_S"] + ex["tirosh_G2M"]); subsets["mitochondrial (MT-)"] = np.array([i for i, g in enumerate(sct) if g.startswith("MT-")])
subsets["ribosomal (RPS/RPL)"] = np.array([i for i, g in enumerate(sct) if g.startswith(("RPS", "RPL"))]); subsets["genes detected in >=50% of cells"] = np.flatnonzero(frac_det >= 0.5)
for nm, ix in subsets.items():
    if len(ix) >= 5: rep_mat("A Pearson, SCT data (log), gene subset", f"{nm} (n={len(ix)})", pearson_rows(D[:, ix].copy()))
# ---- B: rank / cosine variants
for nm in ("3,000 SCT variable genes", "top 5000 by variance"):
    ix = subsets[nm]; rep_mat("B Spearman, SCT data (log)", nm, spearman_rows(D[:, ix])); rep_mat("B cosine, SCT data (log)", nm, cosine_rows(D[:, ix]))
rep_mat("A Pearson, SCT data (log), gene subset", f"all genes (n={len(sct)})", pearson_rows(D))      # in place: D is destroyed
del D
# ---- C: scale.data (Pearson residuals)
with h5py.File(H5) as f: S = f["obsm/SCT_scale_data"][cols.tolist()].astype(np.float32)
rep_mat("C scale.data (Pearson residuals), 3,000 genes", "Pearson", pearson_rows(S.copy())); rep_mat("C scale.data (Pearson residuals), 3,000 genes", "cosine", cosine_rows(S)); rep_mat("C scale.data (Pearson residuals), 3,000 genes", "Spearman", spearman_rows(S))
# ---- D: embeddings
pc = pd.read_csv(f"{EXP}/red_pca.csv", index_col=0).iloc[:, :50].to_numpy(np.float64)[m]
for k in (2, 5, 10, 20, 50):
    rep_mat("D embedding: PCs, Pearson across PCs", f"first {k} PCs", np.corrcoef(pc[:, :k])) if k >= 3 else None
    rep("D embedding: PCs, negative Euclidean distance", f"first {k} PCs", pdist_sim(pc[:, :k], a, b), pdist_sim(pc[:, :k], ra, rb))
for nm in ("umap_rna", "umap_rpca"):
    u = pd.read_csv(f"{EXP}/red_{nm}.csv", index_col=0).to_numpy(np.float64)[m]; rep("D embedding: UMAP, negative Euclidean distance", nm, pdist_sim(u, a, b), pdist_sim(u, ra, rb))
# ---- E / F: neighbour graphs (cells x cells, SCT_nn = k-nearest neighbours, SCT_snn = shared nearest neighbours)
def load_graph(name):
    n = 76451; p = np.fromfile(f"{EXP}/graph_{name}_p.bin", dtype=np.float64).astype(np.int64); i = np.fromfile(f"{EXP}/graph_{name}_i.bin", dtype=np.int32); x = np.fromfile(f"{EXP}/graph_{name}_x.bin", dtype=np.float64)
    return sp.csc_matrix((x, i, p), shape=(n, n)).tocsr()
for g in ("SCT_snn", "SCT_nn"):
    G = load_graph(g); Gs = G[cols][:, cols].tocsr(); Gs = Gs.maximum(Gs.T).tocsr()
    rep("E graph weight (sym.), SCT graphs", g + " weight", np.asarray(Gs[a, b]).ravel(), np.asarray(Gs[ra, rb]).ravel())
    rep("E graph connected (0/1), SCT graphs", g + " is a neighbour", (np.asarray(Gs[a, b]).ravel() > 0).astype(float), (np.asarray(Gs[ra, rb]).ravel() > 0).astype(float))
    if g == "SCT_snn":
        W = sp.diags(1.0 / np.maximum(np.asarray(Gs.sum(1)).ravel(), 1e-9)) @ Gs
        for hops in (1, 2):
            Sm = S.copy()
            for _ in range(hops): Sm = 0.5 * Sm + 0.5 * (W @ Sm)
            rep_mat("F neighbour-smoothed scale.data (SNN, mix 0.5), Pearson", f"{hops} step(s)", pearson_rows(Sm.copy()))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/figA_battery_hESC_p2.csv", index=False); pd.set_option("display.width", 250, "display.max_rows", 200)
print("\npaper Fig 2A: between ~0.55, within ~0.68, d = 3.51\n"); print(T.sort_values("cohens_d", ascending=False).round(3).to_string(index=False))
# ---- plot: Cohen's d per measure
Ts = T.sort_values("cohens_d"); BLUE, INK, MUTED, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
fig, ax = plt.subplots(figsize=(11, max(6, 0.22 * len(Ts) + 1.8)), facecolor=SURF); ax.set_facecolor(SURF)
y = np.arange(len(Ts)); ax.hlines(y, 0, Ts.cohens_d, color=BLUE, linewidth=1.4, alpha=0.8); ax.plot(Ts.cohens_d, y, "o", color=BLUE, markersize=4.5)
ax.axvline(3.51, color=MUTED, linestyle=(0, (3, 3)), linewidth=1.2); ax.text(3.45, 1, "paper: d = 3.51", ha="right", color=MUTED, fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([f"{g.split(' ', 1)[0]}  {mname}" for g, mname in zip(Ts.group, Ts.measure)], fontsize=7.2, color=INK); ax.set_xlim(-0.05, 3.7)
ax.set_xlabel("Cohen's d: same-clone pairs vs random different-clone pairs (higher = clone-mates more similar)", color=INK, fontsize=9.5); ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for sp_ in ("top", "right", "left"): ax.spines[sp_].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0, labelsize=8)
ax.set_title("hESCs p2: how far from the paper's Fig 2A effect is each measure?", loc="left", color=INK, fontsize=11.5, pad=10)
fig.subplots_adjust(left=0.42, right=0.98, bottom=0.07, top=0.95)
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/figA_battery_hESC_p2.{ext}", dpi=170, facecolor=SURF)
