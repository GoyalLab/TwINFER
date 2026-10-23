"""Clone pairs vs random pairs with and WITHOUT matching the random pairs' cell types (clust_annot_02), per sample.
unmatched random : random pairs of cells from different clones (any cell types)                         [what the earlier boxplots use]
type-matched     : for each clone pair with cell types (ta, tb), one random pair of cells from DIFFERENT clones with types (ta, tb)
Both random sets have exactly as many pairs as the sample has clone pairs, redrawn 100 times. Wilcoxon rank-sum (two-sided), median p over the draws,
AUC = P(random-pair distance > clone-pair distance). Draw 0 is plotted. Unweighted pairs. p-values are optimistic (pairs of one clone share cells).
usage: pca_clonal_distance_typematched.py N_PCS [PCA_CSV_PATTERN_with_{sample} | -] [TAG] [LABEL]
"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/pca_clonal_distance"
npc = int(sys.argv[1]); PAT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None; TAG = sys.argv[3] if len(sys.argv) > 3 else ""
LABEL = sys.argv[4] if len(sys.argv) > 4 else "stored Seurat PCA (SCT)"
SAMPLES = ["hESCs_p2_parent", "hESCs_p3_same", "hESCs_p3_sis", "PS_sis", "DE_same", "DE_sis"]; NREP = 100
rng = np.random.default_rng(0)
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False)
if PAT is None: P = pd.read_csv(f"{EXP}/red_pca.csv", index_col=0).iloc[:, :npc].to_numpy(np.float64)
def get_Z(s, m): return P[m] if PAT is None else pd.read_csv(f"{EXP}/{PAT.format(sample=s)}", index_col=0).reindex(obs.index[m]).iloc[:, :npc].to_numpy(np.float64)
def pairs_within(l):
    o = np.argsort(l, kind="stable"); ls = l[o]; st = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]]); en = np.r_[st[1:], len(ls)]; A, B = [], []
    for s, e in zip(st, en):
        if e - s >= 2: i, j = np.triu_indices(e - s, 1); A.append(o[s + i]); B.append(o[s + j])
    return np.concatenate(A), np.concatenate(B)
dist = lambda Z, a, b: np.linalg.norm(Z[a] - Z[b], axis=1)

def matched_draw(ta, tb, clone, by_type):
    """for each clone pair (types ta[k], tb[k]) draw a random pair of cells from different clones with types (ta[k], tb[k]); returns index arrays and a keep mask"""
    n = len(ta); ia = np.empty(n, int); ib = np.empty(n, int); ok = np.zeros(n, bool); todo = np.arange(n)
    for _ in range(60):
        if len(todo) == 0: break
        for key in np.unique(np.c_[ta[todo], tb[todo]], axis=0):
            sel = todo[(ta[todo] == key[0]) & (tb[todo] == key[1])]
            ia[sel] = rng.choice(by_type[key[0]], len(sel)); ib[sel] = rng.choice(by_type[key[1]], len(sel))
        good = clone[ia[todo]] != clone[ib[todo]]; ok[todo[good]] = True; todo = todo[~good]
    return ia, ib, ok

rows, plotdata = [], {}
for s in SAMPLES:
    m = (obs["SampleID"].astype(str) == s).values & obs["clone.id"].notna().values
    Z = get_Z(s, m); clone = obs.loc[m, "clone.id"].astype(int).to_numpy(); ann = pd.Categorical(obs.loc[m, "clust_annot_02"].astype(str)).codes
    by_type = {t: np.flatnonzero(ann == t) for t in np.unique(ann)}
    a, b = pairs_within(clone); cd = dist(Z, a, b); N = len(cd); ta, tb = ann[a], ann[b]
    same_clone_pairs = float((ta == tb).mean())
    res = {k: [] for k in ("p_un", "auc_un", "p_ma", "auc_ma", "dropped")}
    for r in range(NREP):
        ra = rng.integers(0, len(Z), 3 * N); rb = rng.integers(0, len(Z), 3 * N); okk = clone[ra] != clone[rb]; ru = dist(Z, ra[okk][:N], rb[okk][:N])
        ia, ib, ok = matched_draw(ta, tb, clone, by_type); rm = dist(Z, ia[ok], ib[ok])
        tu = mannwhitneyu(cd, ru, alternative="two-sided"); res["p_un"].append(tu.pvalue); res["auc_un"].append(1 - tu.statistic / (N * len(ru)))
        # compare against the clone pairs that found a match (all, except a few rare-type pairs)
        tm = mannwhitneyu(cd[ok], rm, alternative="two-sided"); res["p_ma"].append(tm.pvalue); res["auc_ma"].append(1 - tm.statistic / (ok.sum() * len(rm)))
        res["dropped"].append(int((~ok).sum()))
        if r == 0: plotdata[s] = (cd, ru, rm)
    ru0, rm0 = plotdata[s][1], plotdata[s][2]
    rows.append(dict(sample=s, clone_pairs=N, frac_clone_pairs_same_type=same_clone_pairs, frac_unmatched_random_same_type=float(np.mean([ann[i] == ann[j] for i, j in zip(*rng.integers(0, len(Z), (2, 20000)))])),
                     median_clone=np.median(cd), median_unmatched=np.median(ru0), median_matched=np.median(rm0),
                     AUC_unmatched=np.median(res["auc_un"]), p_unmatched_median=np.median(res["p_un"]), p_unmatched_worst=np.max(res["p_un"]),
                     AUC_matched=np.median(res["auc_ma"]), p_matched_median=np.median(res["p_ma"]), p_matched_worst=np.max(res["p_ma"]), unmatched_pairs_dropped=int(np.median(res["dropped"]))))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
T = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); T.to_csv(f"{OUT}/pca{npc}{TAG}_typematched_vs_unmatched.csv", index=False)
pd.set_option("display.width", 250); print(T.round(4).to_string(index=False))

BLUE, ORANGE, AQUA, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
fig, ax = plt.subplots(figsize=(12, 5.4), facecolor=SURF); ax.set_facecolor(SURF); w, gap = 0.26, 0.015; offs = (-(w + gap), 0, w + gap); tops = {}
for k, s in enumerate(SAMPLES):
    cd, ru, rm = plotdata[s]; tops[s] = max(cd.max(), ru.max(), rm.max())
    for off, x, col in zip(offs, (cd, ru, rm), (BLUE, ORANGE, AQUA)):
        ax.boxplot(x, positions=[k + off], widths=w, patch_artist=True, showfliers=True, manage_ticks=False,
                   boxprops=dict(facecolor=col, edgecolor=col, alpha=0.9, linewidth=1), medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=col, linewidth=1),
                   capprops=dict(color=col, linewidth=1), flierprops=dict(marker="o", markersize=1.6, markerfacecolor=col, markeredgecolor="none", alpha=0.25))
    ax.text(k, -0.075, f"n = {len(cd):,} clone pairs", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=MUTED)
star = lambda p: "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
span = max(tops.values()) - ax.get_ylim()[0]
for k, s in enumerate(SAMPLES):
    r = T.set_index("sample").loc[s]
    for lvl, (o2, p) in enumerate(((offs[1], r.p_unmatched_median), (offs[2], r.p_matched_median))):
        y = tops[s] + (0.03 + 0.075 * lvl) * span; x0, x1 = k + offs[0], k + o2
        ax.plot([x0, x0, x1, x1], [y - 0.012 * span, y, y, y - 0.012 * span], color=MUTED, linewidth=1, clip_on=False)
        ax.text((x0 + x1) / 2, y + 0.004 * span, star(p), ha="center", va="bottom", fontsize=10 if star(p) != "ns" else 8, color=INK)
ax.set_ylim(top=max(tops.values()) + 0.2 * span)
ax.set_xticks(range(len(SAMPLES))); ax.set_xticklabels(SAMPLES, fontsize=10, color=INK); ax.set_ylabel(f"Euclidean distance in PCA space ({npc} PCs)", color=INK)
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0)
ax.set_title("Random pairs matched vs not matched on cell type: distance in PCA space, within each sample", loc="left", color=INK, fontsize=11.5, pad=26)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA)], labels=["same clone", "random pairs, any cell types", "random pairs, same cell-type combination as each clone pair"],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK)
fig.text(0.07, 0.05, "PCA: " + LABEL, fontsize=9, color=MUTED, ha="left", va="bottom")
fig.text(0.07, 0.022, "Stars: Wilcoxon rank-sum vs same clone, equal pair numbers, two-sided, median p of 100 draws (lower bracket: any cell types; upper bracket: type-matched).", fontsize=7.8, color=MUTED, ha="left", va="bottom")
fig.text(0.07, 0.004, "* p<0.05   ** p<0.01   *** p<0.001   **** p<0.0001", fontsize=7.8, color=MUTED, ha="left", va="bottom")
fig.subplots_adjust(bottom=0.2, top=0.86, left=0.07, right=0.985)
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/pca{npc}{TAG}_typematched_vs_unmatched_boxplot.{ext}", dpi=200, facecolor=SURF)
print("saved", f"{OUT}/pca{npc}{TAG}_typematched_vs_unmatched_boxplot.png")
