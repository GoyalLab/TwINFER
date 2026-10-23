"""Wilcoxon rank-sum (Mann-Whitney U) test: same-clone pair distances vs an EQUAL number of random different-clone pair distances, per sample.
random pairs are redrawn 100 times (each draw has exactly as many pairs as the sample has clone pairs); reported: median and range over the draws.
Unweighted pairs (every within-clone pair counts once). NOTE: pairs of one clone share cells, so they are not independent and the p-values are optimistic;
the clone-label permutation test (pca_clonal_distance.py) is the valid one.
usage: pca_clonal_distance_wilcoxon.py N_PCS [PCA_CSV_PATTERN_with_{sample} | -] [TAG]
"""
import sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/pca_clonal_distance"
npc = int(sys.argv[1]); PAT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None; TAG = sys.argv[3] if len(sys.argv) > 3 else ""
rng = np.random.default_rng(0); NREP = 100
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False)
if PAT is None: P = pd.read_csv(f"{EXP}/red_pca.csv", index_col=0).iloc[:, :npc].to_numpy(np.float64)
def get_Z(s, m): return P[m] if PAT is None else pd.read_csv(f"{EXP}/{PAT.format(sample=s)}", index_col=0).reindex(obs.index[m]).iloc[:, :npc].to_numpy(np.float64)
def pairs_within(l):
    o = np.argsort(l, kind="stable"); ls = l[o]; st = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]]); en = np.r_[st[1:], len(ls)]; A, B = [], []
    for s, e in zip(st, en):
        if e - s >= 2: i, j = np.triu_indices(e - s, 1); A.append(o[s + i]); B.append(o[s + j])
    return np.concatenate(A), np.concatenate(B)
dist = lambda Z, a, b: np.linalg.norm(Z[a] - Z[b], axis=1)
rows = []
for s in ["hESCs_p2_parent", "hESCs_p3_same", "hESCs_p3_sis", "PS_sis", "DE_same", "DE_sis"]:
    m = (obs["SampleID"].astype(str) == s).values & obs["clone.id"].notna().values
    Z = get_Z(s, m); clone = obs.loc[m, "clone.id"].astype(int).to_numpy(); a, b = pairs_within(clone); cd = dist(Z, a, b); N = len(cd)
    p2, p1, auc, dmed = [], [], [], []
    for _ in range(NREP):
        ra = rng.integers(0, len(Z), 3 * N); rb = rng.integers(0, len(Z), 3 * N); ok = clone[ra] != clone[rb]
        rd = dist(Z, ra[ok][:N], rb[ok][:N]); assert len(rd) == N
        r2 = mannwhitneyu(cd, rd, alternative="two-sided"); r1 = mannwhitneyu(cd, rd, alternative="less")
        p2.append(r2.pvalue); p1.append(r1.pvalue); auc.append(1 - r2.statistic / (N * N)); dmed.append(np.median(cd) - np.median(rd))   # AUC = P(random > clone)
    q = lambda x: (np.median(x), np.min(x), np.max(x))
    rows.append(dict(sample=s, pairs_each_group=N, median_clone=np.median(cd), median_diff_clone_minus_random=q(dmed)[0], AUC_random_gt_clone=q(auc)[0],
                     p_two_sided_median=q(p2)[0], p_two_sided_max=q(p2)[2], p_one_sided_clone_closer_median=q(p1)[0], p_one_sided_max=q(p1)[2]))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/pca{npc}{TAG}_wilcoxon_equal_n.csv", index=False)
pd.set_option("display.width", 220); print(T.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
