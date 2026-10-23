"""PCA distance: cells of the same clone vs random cells, within each sample (Novo Nordisk hPSC endoderm).
Uses the Seurat PCA (50 PCs, SCT assay) exported by export_seurat.R. Only cells with a clone.id.
Per sample:
  - clone pairs   : all within-clone cell pairs
  - random pairs  : 200,000 random pairs of cells from DIFFERENT clones (same sample)
  - AUC           : P(random-pair distance > clone-pair distance) (0.5 = no clonal similarity)
  - permutation   : clone labels shuffled across the sample's cells (clone sizes kept), 1,000 draws; z and one-sided p of the mean clone-pair distance
  - same-type control: the same comparison restricted to pairs with the same clust_annot_02 label (permutation within annotation)
usage: pca_clonal_distance.py [N_PCS]   (default 50; also run 10)
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
OUT = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/pca_clonal_distance"
npc = int(sys.argv[1]) if len(sys.argv) > 1 else 50
rng = np.random.default_rng(0)
PAT = sys.argv[2] if len(sys.argv) > 2 else None                    # e.g. pca_log1p_cp10k_hvg5000_{sample}.csv inside export/ = separate PCA per sample
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False)
if PAT is None:
    pca = pd.read_csv(f"{EXP}/red_pca.csv", index_col=0); assert (pca.index.values == obs.index.values).all()
    P = pca.iloc[:, :npc].to_numpy(np.float64)
def get_Z(sample, m):
    if PAT is None: return P[m]
    d = pd.read_csv(f"{EXP}/{PAT.format(sample=sample)}", index_col=0)
    return d.reindex(obs.index[m]).iloc[:, :npc].to_numpy(np.float64)

def pairs_within(labels):
    """all within-group index pairs; groups = equal labels"""
    o = np.argsort(labels, kind="stable"); ls = labels[o]
    st = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]]); en = np.r_[st[1:], len(ls)]
    A, B = [], []
    for s, e in zip(st, en):
        if e - s >= 2:
            i, j = np.triu_indices(e - s, 1); A.append(o[s + i]); B.append(o[s + j])
    return (np.concatenate(A), np.concatenate(B)) if A else (np.array([], int), np.array([], int))

def dist(Z, a, b): return np.linalg.norm(Z[a] - Z[b], axis=1)

def auc(clone_d, rand_d):                                           # P(random > clone), ties 0.5
    r = rankdata(np.r_[rand_d, clone_d]); n1, n2 = len(rand_d), len(clone_d)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)

def perm_mean(Z, clone, nperm, strata=None):
    """mean clone-pair distance under clone-label permutation (within strata if given); returns observed, null array"""
    a, b = pairs_within(clone); obs_m = dist(Z, a, b).mean(); nulls = np.empty(nperm)
    idx_by = [np.arange(len(clone))] if strata is None else [np.flatnonzero(strata == k) for k in np.unique(strata)]
    for t in range(nperm):
        lab = clone.copy()
        for ix in idx_by: lab[ix] = clone[ix][rng.permutation(len(ix))]
        aa, bb = pairs_within(lab); nulls[t] = dist(Z, aa, bb).mean()
    return obs_m, nulls

rows = []
for s in ["hESCs_p2_parent", "hESCs_p3_same", "hESCs_p3_sis", "PS_sis", "DE_same", "DE_sis"]:
    m = (obs["SampleID"].astype(str) == s).values & obs["clone.id"].notna().values
    Z = get_Z(s, m); clone = obs.loc[m, "clone.id"].astype(int).to_numpy(); ann = obs.loc[m, "clust_annot_02"].astype(str).to_numpy()
    a, b = pairs_within(clone); cd = dist(Z, a, b)
    ra = rng.integers(0, len(Z), 600_000); rb = rng.integers(0, len(Z), 600_000); ok = clone[ra] != clone[rb]
    rd = dist(Z, ra[ok][:200_000], rb[ok][:200_000])
    o1, n1 = perm_mean(Z, clone, 1000)
    # same-annotation control: only pairs with the same clust_annot_02, permutation of clone labels within annotation
    same = ann[a] == ann[b]; cd_same = cd[same]
    rs_ok = ok & (ann[ra] == ann[rb]); rd_same = dist(Z, ra[rs_ok][:200_000], rb[rs_ok][:200_000])
    o2, n2 = perm_mean(Z, clone, 1000, strata=ann)
    rows.append(dict(sample=s, cells=len(Z), clones_ge2=len(np.unique(clone[np.isin(clone, np.unique(clone, return_counts=True)[0][np.unique(clone, return_counts=True)[1] >= 2])])), clone_pairs=len(cd),
                     median_clone=np.median(cd), median_random=np.median(rd), ratio=np.median(cd) / np.median(rd), AUC=auc(cd, rd),
                     perm_z=(o1 - n1.mean()) / n1.std(), perm_p=(1 + (n1 <= o1).sum()) / 1001,
                     frac_pairs_same_type=same.mean(), AUC_same_type=auc(cd_same, rd_same), perm_z_same_type=(o2 - n2.mean()) / n2.std(), perm_p_same_type=(1 + (n2 <= o2).sum()) / 1001))
    print(rows[-1], flush=True)
import os; os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{OUT}/pca{npc}{TAG}_clone_vs_random.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
