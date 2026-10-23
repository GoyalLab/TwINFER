"""Reproduce the clone-level numbers of Linneberg-Agerholm et al. (bioRxiv 2026.09.02.748863), Fig 2, from the exported Seurat object.
 A (Fig 2A)  Pearson correlation between cell pairs, same clone vs different clone, at hESCs p2 (paper: Cohen's d = 3.51). Normalization/gene set not stated -> several variants.
 I (Fig 2I)  clone purity (fraction of a clone's cells in its dominant cluster) at DE vs a clone-size-matched weighted-random null (paper: 0.68 vs 0.646, n = 495 clones).
 J (Fig 2J)  dominant-cluster agreement between same-well and sister-well, hESCs p3 (paper: 0.77 vs 0.545, n = 343 clones) and DE (0.687 vs 0.50, n = 67 clones).
Clone filters (paper Suppl 2A) are not available; approximations are printed. Timepoints: p2 = hESCs_p2_parent, p3 = hESCs_p3_same + hESCs_p3_sis, PS = PS_sis, DE = DE_same + DE_sis.
"""
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, binomtest

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False); obs["clone"] = obs["clone.id"]
TP = {"hESCs_p2_parent": "p2", "hESCs_p3_same": "p3", "hESCs_p3_sis": "p3", "PS_sis": "PS", "DE_same": "DE", "DE_sis": "DE"}; obs["tp"] = obs.SampleID.astype(str).map(TP)
bar = obs[obs.clone.notna()].copy(); bar["clone"] = bar.clone.astype(int); bar["cl"] = bar.clust_annot_02.astype(str)

# ---------------------------------------------------------------- A: Pearson within vs between clone, hESCs p2
def load_dense(prefix, cols, nr):
    p = np.fromfile(f"{EXP}/{prefix}_p.bin", dtype=np.float64).astype(np.int64)
    I = np.memmap(f"{EXP}/{prefix}_i.bin", dtype=np.int32, mode="r"); X = np.memmap(f"{EXP}/{prefix}_x.bin", dtype=np.float64, mode="r")
    D = np.zeros((len(cols), nr), np.float32)
    for s in range(0, len(cols), 2000):
        c = cols[s:s + 2000]; l = p[c + 1] - p[c]; idx = np.repeat(p[c] - np.r_[0, np.cumsum(l)[:-1]], l) + np.arange(int(l.sum()))
        D[s + np.repeat(np.arange(len(c)), l), np.asarray(I[idx])] = np.asarray(X[idx])
    return D
def pearson_rows(D):                                       # pearson across genes between all cells (dense, in place)
    # D -= D.mean(1, keepdims=True); D /= np.sqrt((D ** 2).mean(1, keepdims=True)); return (D @ D.T) / D.shape[1]     # replaced: (D ** 2) made a full-size temporary copy
    D -= D.mean(1, keepdims=True); D /= np.sqrt(np.einsum("ij,ij->i", D, D) / D.shape[1])[:, None]; return (D @ D.T) / D.shape[1]
def cohen_d(a, b): return (a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
def pairs_within(l):
    o = np.argsort(l, kind="stable"); ls = l[o]; st = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]]); en = np.r_[st[1:], len(ls)]; A, B = [], []
    for s, e in zip(st, en):
        if e - s >= 2: i, j = np.triu_indices(e - s, 1); A.append(o[s + i]); B.append(o[s + j])
    return np.concatenate(A), np.concatenate(B)

allcells = obs.index.to_numpy(); sample = "hESCs_p2_parent"
m = ((obs.SampleID.astype(str) == sample) & obs.clone.notna()).to_numpy(); cols = np.flatnonzero(m); clone = obs.clone[m].astype(int).to_numpy()
a, b = pairs_within(clone); ra = rng.integers(0, len(cols), 600_000); rb = rng.integers(0, len(cols), 600_000); ok = clone[ra] != clone[rb]; ra, rb = ra[ok][:300_000], rb[ok][:300_000]
sct_feat = pd.read_csv(f"{EXP}/SCT_counts_features.csv").feature.astype(str).values; var_feat = set(pd.read_csv(f"{EXP}/SCT_variable_features.csv").feature.astype(str))
rows = []
def report(name, R):
    w, bt = R[a, b], R[ra, rb]; rows.append(dict(variant=name, mean_within=w.mean(), mean_between=bt.mean(), sd_between=bt.std(), cohens_d=cohen_d(w, bt),
                  AUC_within_gt_between=mannwhitneyu(w, bt, alternative="greater").statistic / (len(w) * len(bt)), within_pairs=len(w), between_pairs=len(bt))); print(rows[-1], flush=True)
D = load_dense("SCT_data", cols, len(sct_feat)); keep = np.array([g in var_feat for g in sct_feat])
R = pearson_rows(D[:, keep].copy()); report("SCT data (log), 3,000 variable genes", R); del R
R = pearson_rows(D); report("SCT data (log), all 34,207 SCT genes", R); del R, D
rna_feat_n = int(open(f"{EXP}/RNA_counts_dim.txt").read().split()[0]); D = load_dense("RNA_counts", cols, rna_feat_n); tot = D.sum(1, keepdims=True)
# D = np.log1p(D * (1e4 / tot).astype(np.float32)); R = pearson_rows(D)     # replaced: full-size temporaries
D *= (1e4 / tot).astype(np.float32); np.log1p(D, out=D); R = pearson_rows(D); report("log1p(CP10k) of raw RNA counts, all 38,936 genes", R); del R, D
pc = pd.read_csv(f"{EXP}/red_pca.csv", index_col=0).iloc[:, :50].to_numpy(np.float64)[m]; R = np.corrcoef(pc); report("stored SCT PCA, 50 PCs", R)
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/figA_pearson_within_vs_between_hESC_p2.csv", index=False); print("\nFig 2A (paper: Cohen's d = 3.51)"); print(A.round(3).to_string(index=False))

# ---------------------------------------------------------------- clone filters (approximation of Suppl 2A)
cnt = bar.groupby(["clone", "tp"]).size().unstack(fill_value=0).reindex(columns=["p2", "p3", "PS", "DE"], fill_value=0)
cnt["total"] = cnt.sum(1)
def clones_with(min_per_tp, min_total, need):          # detected at every timepoint in `need` with >= min_per_tp cells, and >= min_total cells overall
    ok = (cnt.total >= min_total)
    for t in need: ok &= cnt[t] >= min_per_tp
    return set(cnt.index[ok])
def dominant(counts):
    mx = counts.max(); return rng.choice(np.flatnonzero(counts == mx))

# ---------------------------------------------------------------- I: purity at DE vs weighted-random null
def purity_test(clones, nperm=1000):
    d = bar[(bar.tp == "DE") & bar.clone.isin(clones)]; cats = sorted(bar[bar.tp == "DE"].cl.unique()); pdist = bar[bar.tp == "DE"].cl.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    obs_p, null_p = [], []
    for c, g in d.groupby("clone"):
        n = len(g); k = g.cl.value_counts().to_numpy(); obs_p.append(k.max() / n)
        s = rng.multinomial(n, pdist, size=nperm); null_p.append((s.max(1) / n).mean())
    obs_p, null_p = np.array(obs_p), np.array(null_p)
    return dict(clones=len(obs_p), observed=obs_p.mean(), weighted_random=null_p.mean(), diff_points=100 * (obs_p.mean() - null_p.mean()), wilcoxon_p=wilcoxon(obs_p - null_p).pvalue)
# ---------------------------------------------------------------- J: dominant-cluster agreement same-well vs sister-well
def agreement(tp, same, sis, clones, mincells, nsim=1000):
    d = bar[(bar.tp == tp) & bar.clone.isin(clones)]; cats = sorted(d.cl.unique()); idx = {c: i for i, c in enumerate(cats)}
    pdist = d.cl.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    cs = d[d.SampleID.astype(str) == same].groupby("clone").cl.agg(list); ci = d[d.SampleID.astype(str) == sis].groupby("clone").cl.agg(list); both = cs.index.intersection(ci.index)
    agree, exp = [], []
    for c in both:
        x, y = cs[c], ci[c]
        if len(x) < mincells or len(y) < mincells: continue
        kx = np.bincount([idx[v] for v in x], minlength=len(cats)); ky = np.bincount([idx[v] for v in y], minlength=len(cats))
        agree.append(dominant(kx) == dominant(ky))
        sx = rng.multinomial(len(x), pdist, size=nsim); sy = rng.multinomial(len(y), pdist, size=nsim)
        dx = np.array([rng.choice(np.flatnonzero(r == r.max())) for r in sx]); dy = np.array([rng.choice(np.flatnonzero(r == r.max())) for r in sy]); exp.append((dx == dy).mean())
    agree, exp = np.array(agree), np.array(exp)
    if len(agree) == 0: return dict(clones=0)
    return dict(clones=len(agree), observed_agreement=agree.mean(), weighted_random=exp.mean(), binom_p=binomtest(int(agree.sum()), len(agree), exp.mean(), alternative="greater").pvalue)

print("\nclone filter variants (cells per timepoint / total / timepoints required):")
variants = {"all clones": (1, 1, []), ">=2 cells at p2 and p3": (2, 1, ["p2", "p3"]), ">=2 cells at p2,p3,PS,DE (winner-like) & >=4 total": (2, 4, ["p2", "p3", "PS", "DE"]),
            ">=2 cells at p2,p3 and >=4 total": (2, 4, ["p2", "p3"])}
PI, JJ = [], []
for nm, (k, tot_, need) in variants.items():
    cl = clones_with(k, tot_, need); r = purity_test(cl); r["filter"] = nm; r["clones_passing_filter"] = len(cl); PI.append(r)
    for mc in (1, 2, 3, 4):
        for tp, same, sis in (("p3", "hESCs_p3_same", "hESCs_p3_sis"), ("DE", "DE_same", "DE_sis")):
            r = agreement(tp, same, sis, cl, mc); r.update(filter=nm, stage=tp, min_cells_per_replicate=mc); JJ.append(r)
PI = pd.DataFrame(PI); JJ = pd.DataFrame(JJ); PI.to_csv(f"{OUT}/figI_purity_DE.csv", index=False); JJ.to_csv(f"{OUT}/figJ_dominant_cluster_agreement.csv", index=False)
pd.set_option("display.width", 250)
print("\nFig 2I purity at DE (paper: observed 0.68 vs weighted random 0.646, n = 495 clones)"); print(PI[["filter", "clones_passing_filter", "clones", "observed", "weighted_random", "diff_points", "wilcoxon_p"]].round(4).to_string(index=False))
print("\nFig 2J agreement (paper: hESCs p3 0.77 vs 0.545, n = 343; DE 0.687 vs 0.50, n = 67)"); print(JJ[["filter", "stage", "min_cells_per_replicate", "clones", "observed_agreement", "weighted_random", "binom_p"]].round(4).to_string(index=False))
