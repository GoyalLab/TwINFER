#!/usr/bin/env python3
"""One figure: normalized (median clonal pair = 1) clonal-vs-random pair-distance boxplots of selected samples, one skinny panel each.
Panels use the saved pair-level CSVs of pair_distance.py, except 'Watermelon naive (merged)' which is recomputed with both naive replicates
as ONE sample (clonal pairs also counted across replicates when they share a clone barcode).
Output: <OUT>/pair_distance/combined_normalized_selected.pdf + combined_normalized_selected_pairs.csv.gz"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_distance as pdist
from pair_distance import PD_DIR, PCA_DIR, PALETTE, mannwhitneyu, get_significance_symbol, plt, sns

PANELS = [  # (title, dataset, sample)
    ("FM01", "FM01", "FM01"),
    ("Watermelon naive", "Watermelon_naive", "MERGE"),
    ("hPSC T0", "hPSC_endo_yscher", "T0"),
    ("LARRY day 2", "LARRY", "day2"),
    ("CellTag3 day 2", "CellTag3", "day2"),
    ("DTT mouse exp1", "DTT_mouse", "exp1"),
    ("PeTrace E9.5", "PeTrace_E9.5", "E9.5"),
    ("Michaels T1", "Michaels_T1T2", "T1"),
    ("SpaceBar section 2", "SpaceBar", "Section2"),
]

# (previous author-name labels kept in combined_normalized_figure.py.before_labels2.bak)
LABELS = {   # x-axis label per panel (key = PANELS title); the saved pair CSV keeps the short titles. [x] = citation placeholder, trailing 1 = dataset number within that publication
    "FM01": "Melanoma patient cell line (treated with Vemurafenib), [x] 1",
    "Watermelon naive": "Breast cancer patient cell line (untreated), [x]",
    "hPSC T0": "hPSC (timepoint 0)",
    "LARRY day 2": "Mouse LSK+ cells (day 2), [x]",
    "CellTag3 day 2": "Mouse LSK+ cells (day 2.5), [x]",
    "DTT mouse exp1": "E13.5 mouse embryo, [x]",
    "PeTrace E9.5": "E9.5 mouse embryo, PeTrace, [x]",
    "Michaels T1": "T-cell progenitors, [x]",
    "SpaceBar section 2": "Melanoma xenografts, [x]",
}


def merged_watermelon():
    z = np.load(f"{PCA_DIR}/Watermelon_naive_pca5k.npz", allow_pickle=True)
    cells, P = z["cells"].astype(str), z["pca"].astype(np.float32)
    m = pdist.meta("Watermelon_naive").reindex(cells)
    clones = m["clone"].astype(object).to_numpy()
    idx = np.arange(len(cells))
    import json
    pdist.KE = max(5, int(json.load(open(f"{PCA_DIR}/summary_Watermelon_naive.json"))["elbow_pc"]))   # only feeds the secondary elbow-K column
    s, pairs = pdist.group_stats(P, P.shape[1], idx, clones, "ALL")
    return s, pairs


rows, info = [], []
for title, ds, smp in PANELS:
    if smp == "MERGE":
        s, d = merged_watermelon()
        d = d.copy()
    else:
        pf = pd.read_csv(f"{PD_DIR}/{ds}_pair_distances.csv.gz", keep_default_na=False)
        d = pf[(pf["sample"] == smp) & (pf["celltype"] == "ALL")].copy()
        s = pd.read_csv(f"{PD_DIR}/{ds}_pair_distance_summary.csv", keep_default_na=False)
        s = s[(s["sample"] == smp) & (s["celltype"] == "ALL")].iloc[0].to_dict()
    # normalize by the clonal median of the pairs actually plotted (all 50 PCs)
    mc = d.loc[d["pair_type"] == "clonal", "dist_K"].median()
    d["dist_norm"] = d["dist_K"] / mc
    d["panel"] = title.replace("\n", " ")
    rows.append(d[["panel", "pair_type", "dist_K", "dist_norm"]])
    info.append((title, int(s["n_pairs_used"]), int(s["n_clonal_pairs_available"]), d))

out = pd.concat(rows)
TAG = os.environ.get("COMBINED_TAG", "")      # optional suffix so a rerun does not overwrite the earlier figure
# out.to_csv(f"{PD_DIR}/combined_normalized_selected_pairs.csv.gz", index=False)   # replaced: tag-aware version below
out.to_csv(f"{PD_DIR}/combined_normalized_selected_pairs{TAG}.csv.gz", index=False)

# per-panel upper whisker (Q3 + 1.5 IQR capped at max, over both boxes) -> bracket sits right above that panel's boxes
def whisker_top(d):
    t = 0.0
    for _, g in d.groupby("pair_type"):
        q1, q3 = np.percentile(g["dist_norm"], [25, 75])
        t = max(t, min(q3 + 1.5 * (q3 - q1), g["dist_norm"].max()))
    return t


tops = [whisker_top(d) for _, _, _, d in info]
ymax = max(tops) * 1.18            # shared axis; room for the highest bracket + stars

n = len(info)
# fig, axes = plt.subplots(1, n, figsize=(0.95 * n + 1.2, 5.0), sharey=True)   # replaced: wider panels for the long labels
# fig, axes = plt.subplots(1, n, figsize=(1.3 * n + 1.2, 5.6), sharey=True)   # replaced: wider panels, larger label font
fig, axes = plt.subplots(1, n, figsize=(1.9 * n + 1.2, 6.2), sharey=True)
for ax, (title, npairs, navail, d), t in zip(axes, info, tops):
    dd = d.assign(pair_type=d["pair_type"].map({"clonal": "clonal pairs", "random": "random pairs"}))
    sns.boxplot(data=dd, x="pair_type", y="dist_norm", hue="pair_type", order=list(PALETTE), hue_order=list(PALETTE),
                palette=PALETTE, width=0.8, showfliers=False, linewidth=0.8, ax=ax)
    sns.despine(ax=ax, right=True, top=True)
    c = dd[dd["pair_type"] == "clonal pairs"]["dist_norm"]; r = dd[dd["pair_type"] == "random pairs"]["dist_norm"]
    p = mannwhitneyu(c, r, alternative="two-sided").pvalue
    ax.set_ylim(0, ymax)
    h = t * 1.05                                   # bracket right above this panel's boxes
    ax.plot([0, 0, 1, 1], [h, h + ymax * 0.02, h + ymax * 0.02, h], lw=1.0, color="black")
    ax.text(0.5, h + ymax * 0.02, get_significance_symbol(p), ha="center", va="bottom", fontsize=9)
    ax.set_xticks([])
    # ax.set_xlabel(f"{title}\n{npairs:,} pairs" + (f"\n(of {navail:,})" if navail > npairs else ""), fontsize=8, labelpad=6)      # replaced: descriptive labels
    import textwrap
    # ax.set_xlabel(textwrap.fill(LABELS.get(title, title), 19) + f"\n{npairs:,} pairs" + (f" (of {navail:,})" if navail > npairs else ""), fontsize=8, labelpad=6)   # replaced: larger font
    ax.set_xlabel(textwrap.fill(LABELS.get(title, title), 22) + f"\n{npairs:,} pairs" + (f" (of {navail:,})" if navail > npairs else ""), fontsize=11, labelpad=6)      # sample name at the bottom
    ax.set_ylabel("Transcriptomic distance\n(median clonal pair = 1)" if ax is axes[0] else "")
    ax.tick_params(axis="y", labelsize=12)
    if ax is not axes[0]:
        ax.spines["left"].set_visible(False); ax.tick_params(left=False)
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.8) for c in PALETTE.values()]
fig.legend(handles, list(PALETTE), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
for artist in fig.findobj():                       # no clips/masks (Illustrator-editable)
    if hasattr(artist, "set_clip_on"):
        artist.set_clip_on(False)
plt.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.4)
# fig.savefig(f"{PD_DIR}/combined_normalized_selected.pdf", ...)   # replaced: tag-aware version below
fig.savefig(f"{PD_DIR}/combined_normalized_selected{TAG}.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
print("saved", f"{PD_DIR}/combined_normalized_selected{TAG}.pdf")
