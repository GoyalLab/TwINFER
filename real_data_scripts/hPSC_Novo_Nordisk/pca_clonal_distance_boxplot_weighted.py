"""Clone-weighted version of pca_clonal_distance_boxplot.py (each clone counts once, as in the TwinScore runs).
same-clone pairs : all within-clone pairs of a sample, pair weight 1/P_c, P_c = n_c(n_c-1)/2   (clones with n_c >= 2 in that sample)
random pairs     : 200,000 pairs of cells from different clones; cells drawn with probability proportional to 1/n_c (clone-weighted), seed 0
box / whiskers   : weighted quartiles; whiskers = most extreme data within 1.5 x weighted IQR; fliers = all pairs beyond (drawn unweighted)
also writes weighted quantiles, weighted medians, weighted AUC = P(random-pair distance > clone-pair distance), clone-weighted.
usage: pca_clonal_distance_boxplot_weighted.py N_PCS [PCA_CSV_PATTERN_with_{sample}] [TAG] [LABEL]
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
OUT = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/pca_clonal_distance"
npc = int(sys.argv[1]) if len(sys.argv) > 1 else 50
PAT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
LABEL = sys.argv[4] if len(sys.argv) > 4 else "stored Seurat PCA (SCT)"
SAMPLES = ["hESCs_p2_parent", "hESCs_p3_same", "hESCs_p3_sis", "PS_sis", "DE_same", "DE_sis"]
rng = np.random.default_rng(0)
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False)
if PAT is None:
    pca = pd.read_csv(f"{EXP}/red_pca.csv", index_col=0); assert (pca.index.values == obs.index.values).all(); P = pca.iloc[:, :npc].to_numpy(np.float64)
def get_Z(sample, m):
    if PAT is None: return P[m]
    return pd.read_csv(f"{EXP}/{PAT.format(sample=sample)}", index_col=0).reindex(obs.index[m]).iloc[:, :npc].to_numpy(np.float64)

def pairs_within(labels):
    o = np.argsort(labels, kind="stable"); ls = labels[o]
    st = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]]); en = np.r_[st[1:], len(ls)]; A, B, W = [], [], []
    for s, e in zip(st, en):
        n = e - s
        if n >= 2:
            i, j = np.triu_indices(n, 1); A.append(o[s + i]); B.append(o[s + j]); W.append(np.full(len(i), 1.0 / (n * (n - 1) / 2)))
    return np.concatenate(A), np.concatenate(B), np.concatenate(W)
dist = lambda Z, a, b: np.linalg.norm(Z[a] - Z[b], axis=1)

def wquant(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, c, x)
def bxp_stats(x, w):
    q1, med, q3 = wquant(x, w, [0.25, 0.5, 0.75]); iqr = q3 - q1
    lo, hi = x[x >= q1 - 1.5 * iqr].min(), x[x <= q3 + 1.5 * iqr].max()
    return dict(q1=q1, med=med, q3=q3, whislo=lo, whishi=hi, fliers=x[(x < lo) | (x > hi)]), (q1, med, q3)
def wauc(cd, wd, rd):
    rs = np.sort(rd); gt = len(rs) - np.searchsorted(rs, cd, side="right"); eq = np.searchsorted(rs, cd, side="right") - np.searchsorted(rs, cd, side="left")
    return float(np.average((gt + 0.5 * eq) / len(rs), weights=wd))

data, rows = {}, []
for s in SAMPLES:
    m = (obs["SampleID"].astype(str) == s).values & obs["clone.id"].notna().values
    Z = get_Z(s, m); clone = obs.loc[m, "clone.id"].astype(int).to_numpy()
    a, b, wp = pairs_within(clone); cd = dist(Z, a, b)
    _, inv, cnt = np.unique(clone, return_inverse=True, return_counts=True); pc = (1.0 / cnt[inv]); pc /= pc.sum()
    ra = rng.choice(len(Z), 600_000, p=pc); rb = rng.choice(len(Z), 600_000, p=pc); ok = clone[ra] != clone[rb]
    rd = dist(Z, ra[ok][:200_000], rb[ok][:200_000])
    data[s] = (cd, wp, rd)
    for nm, x, w in (("clone (weighted)", cd, wp), ("random (clone-weighted draw)", rd, np.ones(len(rd))), ("clone (unweighted)", cd, np.ones(len(cd)))):
        q = wquant(x, w, [0.05, 0.25, 0.5, 0.75, 0.95]); rows.append((s, nm, len(x), *q))
    rows.append((s, "weighted AUC (P random > clone)", len(cd), wauc(cd, wp, rd), np.nan, np.nan, np.nan, np.nan))

BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURF); ax.set_facecolor(SURF)
w, gap = 0.34, 0.02
for k, s in enumerate(SAMPLES):
    cd, wp, rd = data[s]
    for off, (x, ww), col in ((-(w / 2 + gap / 2), (cd, wp), BLUE), ((w / 2 + gap / 2), (rd, np.ones(len(rd))), ORANGE)):
        st, _ = bxp_stats(x, ww)
        ax.bxp([st], positions=[k + off], widths=w, patch_artist=True, showfliers=False, manage_ticks=False,
               boxprops=dict(facecolor=col, edgecolor=col, alpha=0.9, linewidth=1), medianprops=dict(color=INK, linewidth=1.6),
               whiskerprops=dict(color=col, linewidth=1), capprops=dict(color=col, linewidth=1))
        fl = st["fliers"]
        if len(fl) > 20000: fl = rng.choice(fl, 20000, replace=False)
        ax.scatter(np.full(len(fl), k + off), fl, s=1.6, color=col, alpha=0.25, linewidths=0, zorder=1)
    ax.text(k, -0.075, f"n = {len(cd):,} clone pairs", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=MUTED)
ax.set_xticks(range(len(SAMPLES))); ax.set_xticklabels(SAMPLES, fontsize=10, color=INK)
ax.set_ylabel(f"Euclidean distance in PCA space ({npc} PCs)", color=INK)
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0)
ax.set_title("Clone-weighted: cells of the same clone vs random cells, distance in PCA space, within each sample", loc="left", color=INK, fontsize=11.5, pad=26)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["same clone (each clone counts once: pair weight 1/P)", "random cells from different clones (cells drawn with weight 1/clone size)"],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
fig.text(0.075, 0.035, "PCA: " + LABEL, fontsize=9, color=MUTED, ha="left", va="bottom")
fig.subplots_adjust(bottom=0.2, top=0.86, left=0.075, right=0.985)
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/pca{npc}{TAG}_clone_vs_random_boxplot_weighted.{ext}", dpi=200, facecolor=SURF)
q = pd.DataFrame(rows, columns=["sample", "what", "n", "p5_or_AUC", "p25", "median", "p75", "p95"])
q.to_csv(f"{OUT}/pca{npc}{TAG}_clone_vs_random_weighted_quantiles.csv", index=False)
print(q[q.what.isin(["weighted AUC (P random > clone)"])][["sample", "p5_or_AUC"]].rename(columns={"p5_or_AUC": "weighted_AUC"}).round(3).to_string(index=False))
print("saved", f"{OUT}/pca{npc}{TAG}_clone_vs_random_boxplot_weighted.png")
