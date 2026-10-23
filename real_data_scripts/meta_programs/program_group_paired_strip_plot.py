#!/usr/bin/env python3
"""Paired strip plot of the Gavish et al. 2023 meta-program groups (grouping of panel b): per dataset (FM01, FM06) two categories, Unshared and Shared; y = % of the top-1000 pairs of that list whose target is in the group
(ChIP-Atlas candidate edges, Stage I, no gene filter; counts from fm01_fm06_program_group_unshared_share.csv); one dot per group and list, a line joins the Unshared and Shared dot of the same group.
Line colour = the side with the higher percentage (green: shared, purple: unshared; grey when the two differ by less than 0.5 percentage points).  Group name and percentage at both ends of each line.
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/fm01_fm06_program_group_paired_strip.{pdf,png}.   usage: program_group_paired_strip_plot.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; N = 1000
T = pd.read_csv(f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_unshared_share.csv"); T["u"] = 100 * T.n_unshared / N; T["s"] = 100 * T.n_shared / N
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
CU, CS, CG = "#6a54d3", "#1fb07b", "#9a9a9a"
def dodge(y, gap):                                   # spread label positions so that neighbours are at least `gap` apart (keeps the order)
    o = np.argsort(y); z = np.array(y, float)[o]
    for _ in range(200):
        moved = False
        for i in range(1, len(z)):
            if z[i] - z[i - 1] < gap - 1e-9: m = (z[i] + z[i - 1]) / 2; z[i - 1] = m - gap / 2; z[i] = m + gap / 2; moved = True
        if not moved: break
    out = np.empty_like(z); out[o] = z; return out
fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), sharey=True); fig.patch.set_alpha(0); YM = 22
for ax, ds in zip(axes, ("FM01", "FM06")):
    d = T[T.dataset == ds].reset_index(drop=True); ax.set_facecolor("none")
    for r in d.itertuples():
        diff = r.s - r.u; c = CG if abs(diff) < 0.5 else (CS if diff > 0 else CU)
        ax.plot([0, 1], [r.u, r.s], "-", color=c, lw=2.4, alpha=0.9, zorder=2, solid_capstyle="round")
    ax.scatter(np.zeros(len(d)), d.u, s=70, color=CU, edgecolor="white", linewidth=0.9, zorder=4); ax.scatter(np.ones(len(d)), d.s, s=70, color=CS, edgecolor="white", linewidth=0.9, zorder=4)
    yl = dodge(d.u.to_numpy(), 1.05); yr = dodge(d.s.to_numpy(), 1.05)
    for i, r in enumerate(d.itertuples()):
        ax.text(-0.07, yl[i], f"{r.group}  {r.u:.1f}%", ha="right", va="center", fontsize=9.6, color="#222"); ax.text(1.07, yr[i], f"{r.s:.1f}%  {r.group}", ha="left", va="center", fontsize=9.6, color="#222")
    ax.set_xlim(-0.85, 1.95); ax.set_ylim(-0.5, YM); ax.set_xticks([0, 1]); ax.set_xticklabels(["Unshared list", "Shared list"], fontsize=11.5)
    for lab, c in zip(ax.get_xticklabels(), (CU, CS)): lab.set_color(c); lab.set_fontweight("bold")
    ax.set_title(ds, fontsize=13, fontweight="bold", loc="left"); ax.tick_params(axis="x", length=0)
    for s_ in ("top", "right", "bottom"): ax.spines[s_].set_visible(False)
    ax.spines["left"].set_visible(False) if ds == "FM06" else None; ax.tick_params(labelsize=10)
axes[0].set_ylabel("% of the top-1000 pairs of the list whose target is in the group", fontsize=10.5); axes[1].tick_params(axis="y", length=0, labelleft=False)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0], [0], color=CS, lw=3, label="higher in the shared list"), Line2D([0], [0], color=CU, lw=3, label="higher in the unshared list"), Line2D([0], [0], color=CG, lw=3, label="within 0.5 points")], loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Meta-program groups of the target genes: unshared vs shared list", x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.985)
fig.text(0.01, 0.925, "ChIP-Atlas edges, top 1000 pairs of each list; each line is one meta-program group (Gavish et al. 2023)", fontsize=8.5, color="#666")
fig.tight_layout(rect=[0, 0.05, 1, 0.92], w_pad=0.5)
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_paired_strip"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=140, facecolor="white"); print("saved", out)
