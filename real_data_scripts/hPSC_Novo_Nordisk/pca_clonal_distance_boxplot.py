"""Boxplot of PCA distances (50 PCs): cells of the same clone vs random cells from different clones, per sample.
Same pairs as pca_clonal_distance.py (seed 0): all within-clone pairs, 200,000 random different-clone pairs. One shared y-axis, outliers shown.
Writes PNG + PDF + the plotted distances' summary to analysis_data/hPSC_Novo_Nordisk/pca_clonal_distance/.
usage: pca_clonal_distance_boxplot.py [N_PCS]
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
OUT = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/pca_clonal_distance"
npc = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SAMPLES = ["hESCs_p2_parent", "hESCs_p3_same", "hESCs_p3_sis", "PS_sis", "DE_same", "DE_sis"]
rng = np.random.default_rng(0)
PAT = sys.argv[2] if len(sys.argv) > 2 else None                    # per-sample PCA csv pattern with {sample} (inside export/), default = stored Seurat PCA
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
LABEL = sys.argv[4] if len(sys.argv) > 4 else "stored Seurat PCA (SCT)"
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False)
if PAT is None:
    pca = pd.read_csv(f"{EXP}/red_pca.csv", index_col=0); assert (pca.index.values == obs.index.values).all(); P = pca.iloc[:, :npc].to_numpy(np.float64)
def get_Z(sample, m):
    if PAT is None: return P[m]
    return pd.read_csv(f"{EXP}/{PAT.format(sample=sample)}", index_col=0).reindex(obs.index[m]).iloc[:, :npc].to_numpy(np.float64)

def pairs_within(labels):
    o = np.argsort(labels, kind="stable"); ls = labels[o]
    st = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]]); en = np.r_[st[1:], len(ls)]; A, B = [], []
    for s, e in zip(st, en):
        if e - s >= 2: i, j = np.triu_indices(e - s, 1); A.append(o[s + i]); B.append(o[s + j])
    return np.concatenate(A), np.concatenate(B)
dist = lambda Z, a, b: np.linalg.norm(Z[a] - Z[b], axis=1)

data = {}
for s in SAMPLES:
    m = (obs["SampleID"].astype(str) == s).values & obs["clone.id"].notna().values
    Z = get_Z(s, m); clone = obs.loc[m, "clone.id"].astype(int).to_numpy()
    a, b = pairs_within(clone); cd = dist(Z, a, b)
    ra = rng.integers(0, len(Z), 600_000); rb = rng.integers(0, len(Z), 600_000); ok = clone[ra] != clone[rb]
    data[s] = (cd, dist(Z, ra[ok][:200_000], rb[ok][:200_000]))

BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
TOPS = {}
fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURF); ax.set_facecolor(SURF)
w, gap = 0.34, 0.02
for k, s in enumerate(SAMPLES):
    for off, vals, col in ((-(w / 2 + gap / 2), data[s][0], BLUE), ((w / 2 + gap / 2), data[s][1], ORANGE)):
        bp = ax.boxplot(vals, positions=[k + off], widths=w, patch_artist=True, showfliers=True, manage_ticks=False,
                        boxprops=dict(facecolor=col, edgecolor=col, alpha=0.9, linewidth=1), medianprops=dict(color=INK, linewidth=1.6),
                        whiskerprops=dict(color=col, linewidth=1), capprops=dict(color=col, linewidth=1),
                        flierprops=dict(marker="o", markersize=1.6, markerfacecolor=col, markeredgecolor="none", alpha=0.25))
    TOPS[s] = max(data[s][0].max(), data[s][1].max())
    ax.text(k, -0.075, f"n = {len(data[s][0]):,} clone pairs", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=MUTED)
import os as _os
_wf = f"{OUT}/pca{npc}{TAG}_wilcoxon_equal_n.csv"
if _os.path.exists(_wf):                                             # Wilcoxon rank-sum, clone pairs vs equal number of random pairs (median two-sided p over 100 draws)
    W = pd.read_csv(_wf).set_index("sample"); span = max(TOPS.values()) - ax.get_ylim()[0]
    star = lambda p: "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
    for k, s in enumerate(SAMPLES):
        y = TOPS[s] + 0.03 * span; x0, x1 = k - (w / 2 + gap / 2), k + (w / 2 + gap / 2)
        ax.plot([x0, x0, x1, x1], [y - 0.012 * span, y, y, y - 0.012 * span], color=MUTED, linewidth=1, clip_on=False)
        ax.text(k, y + 0.004 * span, star(W.loc[s, "p_two_sided_median"]), ha="center", va="bottom", fontsize=11 if star(W.loc[s, "p_two_sided_median"]) != "ns" else 9, color=INK)
    ax.set_ylim(top=max(TOPS.values()) + 0.12 * span)
    fig.text(0.075, 0.008, "Wilcoxon rank-sum, clone pairs vs an equal number of random pairs, two-sided, median p of 100 random draws: * p<0.05  ** p<0.01  *** p<0.001  **** p<0.0001", fontsize=8, color=MUTED, ha="left", va="bottom")
ax.set_xticks(range(len(SAMPLES))); ax.set_xticklabels(SAMPLES, fontsize=10, color=INK)
ax.set_ylabel(f"Euclidean distance in PCA space ({npc} PCs)", color=INK)
fig.text(0.075, 0.04, "PCA: " + LABEL, fontsize=9, color=MUTED, ha="left", va="bottom")
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0)
ax.set_title("Cells of the same clone vs random cells: distance in PCA space, within each sample", loc="left", color=INK, fontsize=12, pad=26)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["same clone (all within-clone pairs)", "random cells from different clones (200,000 pairs)"],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
fig.subplots_adjust(bottom=0.2, top=0.86, left=0.075, right=0.985)
import os; os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/pca{npc}{TAG}_clone_vs_random_boxplot.{ext}", dpi=200, facecolor=SURF)
rows = [(s, "clone", len(v), *np.percentile(v, [5, 25, 50, 75, 95])) for s in SAMPLES for v in (data[s][0],)] + \
       [(s, "random", len(v), *np.percentile(v, [5, 25, 50, 75, 95])) for s in SAMPLES for v in (data[s][1],)]
pd.DataFrame(rows, columns=["sample", "pairs", "n", "p5", "p25", "median", "p75", "p95"]).to_csv(f"{OUT}/pca{npc}{TAG}_clone_vs_random_boxplot_quantiles.csv", index=False)
print("saved", f"{OUT}/pca{npc}{TAG}_clone_vs_random_boxplot.png")
