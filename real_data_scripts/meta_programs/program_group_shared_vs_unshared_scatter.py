#!/usr/bin/env python3
"""Scatter of the Gavish et al. 2023 meta-program groups (grouping of panel b): x = % of the top-1000 UNSHARED pairs whose target is in the group, y = % of the top-1000 SHARED pairs whose target is in the group
(ChIP-Atlas candidate edges, Stage I, no gene filter; counts from fm01_fm06_program_group_unshared_share.csv written by program_group_unshared_share_plot.py; both axes are out of 1000 pairs).
One point per group and dataset (FM01 purple, FM06 green); a thin line joins the two datasets of a group; dashed diagonal = same % in both lists (above = more in the shared list, below = more in the unshared list).
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/fm01_fm06_program_group_shared_vs_unshared_scatter.{pdf,png}.   usage: program_group_shared_vs_unshared_scatter.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from adjustText import adjust_text
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; N = 1000
T = pd.read_csv(f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_unshared_share.csv"); T["x"] = 100 * T.n_unshared / N; T["y"] = 100 * T.n_shared / N
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
COL = {"FM01": "#6a54d3", "FM06": "#1fb07b"}; M = 22
fig, ax = plt.subplots(figsize=(7.6, 7.2)); fig.patch.set_alpha(0); ax.set_facecolor("none")
ax.fill_between([0, M], [0, M], [M, M], color="#e9f6f1", zorder=0, lw=0); ax.fill_between([0, M], [0, 0], [0, M], color="#efecfa", zorder=0, lw=0); ax.plot([0, M], [0, M], "--", color="#777", lw=1, zorder=1)
ax.text(1.2, M - 1.3, "more in the shared list", fontsize=10.5, color="#2b8a68", va="top"); ax.text(M - 0.8, 0.9, "more in the unshared list", fontsize=10.5, color="#5a46bd", ha="right", va="bottom")
for g, d in T.groupby("group"): ax.plot(d.x, d.y, "-", color="#bdbdbd", lw=1.2, zorder=2)
for ds in COL:
    d = T[T.dataset == ds]; ax.scatter(d.x, d.y, s=85, color=COL[ds], edgecolor="white", linewidth=0.9, zorder=4, label=ds)
texts = []
for g, d in T.groupby("group"):
    r = d.iloc[d.x.add(d.y).argmax()]; texts.append(ax.text(r.x + 0.5, r.y + 0.5, g, fontsize=10, color="#222", zorder=6))
adjust_text(texts, ax=ax, x=T.x.to_numpy(), y=T.y.to_numpy(), arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5), expand=(1.2, 1.5), ensure_inside_axes=True)
ax.set_xlim(0, M); ax.set_ylim(0, M); ax.set_aspect("equal"); ax.set_xlabel("Unshared list: % of the top-1000 pairs whose target is in the group", fontsize=10.5); ax.set_ylabel("Shared list: % of the top-1000 pairs whose target is in the group", fontsize=10.5)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.tick_params(labelsize=10); ax.legend(loc="upper right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 0.86))
ax.set_title("Meta-program groups of the target genes: shared vs unshared list", fontsize=12.5, loc="left", fontweight="bold", pad=14)
fig.text(0.01, 0.005, "ChIP-Atlas edges, top 1000 of each list; a grey line joins FM01 and FM06 for the same group; dashed line = equal percentage in both lists", fontsize=7.6, color="#666")
fig.tight_layout(rect=[0, 0.02, 1, 1])
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_shared_vs_unshared_scatter"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=140, facecolor="white"); print("saved", out); print(T[["dataset", "group", "x", "y"]].round(1).to_string(index=False))
