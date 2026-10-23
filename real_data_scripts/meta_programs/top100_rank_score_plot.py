#!/usr/bin/env python3
"""Rank-score plot of the top-100 lists of one dataset: x = rank (1 = highest score), y = score; left panel unshared (|z_U|), right panel shared (|z_C|, column h of the Stage I table);
a line joins the points; the top 10 pairs are labelled 'TF-target'.  Lists: analysis_data/fm_meta_programs_comparison/top100_lists/<dataset>_{unshared,shared}.csv (build_top100_lists_alltf.py:
all TF x target Stage I pairs, no gene filter, non-TF regulators excluded before ranking).  Output (Arial, fonttype 42, no clip paths, transparent):
analysis_data/fm_meta_programs_comparison/<dataset>_top100_rank_score.{pdf,png}.   usage: top100_rank_score_plot.py [FM01]"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; L = f"{A}/fm_meta_programs_comparison/top100_lists"
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.fontset": "cm"})
ds = (sys.argv[1:] or ["FM01"])[0]; COL = {"unshared": "#6a54d3", "shared": "#1fb07b"}
PANELS = [("unshared", "z_U", r"Unshared: $|z_{\hat\rho_\Delta}|$"), ("shared", "h", r"Shared: $|z_{\hat\rho_s}|$")]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2)); fig.patch.set_alpha(0)
for ax, (lab, col, title) in zip(axes, PANELS):
    d = pd.read_csv(f"{L}/{ds}_{lab}.csv"); d["score"] = d[col].abs(); d = d.sort_values("score", ascending=False).reset_index(drop=True); d["rank"] = np.arange(1, len(d) + 1)
    ax.plot(d["rank"], d.score, "-", color=COL[lab], lw=1.4, zorder=2); ax.scatter(d["rank"], d.score, s=16, color=COL[lab], edgecolor="white", linewidth=0.4, zorder=3)
    top = d.head(10); ymax, ymin = d.score.max(), d.score.min(); span = ymax - ymin
    for i, r in enumerate(top.itertuples()):                                    # labels stacked in a column to the right of the steep part of the curve; thin leaders to the points
        tx, ty = 40, ymax - span * (0.02 + 0.062 * i)   # (was x=24, step 0.075: the lowest labels sat on the curve)
        ax.annotate(f"{r.rank}. {r.TF}–{r.gene}", xy=(r.rank, r.score), xytext=(tx, ty), fontsize=9, va="center", ha="left", color="#222",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.6, shrinkA=1, shrinkB=2), zorder=4)
    ax.set_xlim(0, 101); ax.set_ylim(ymin - 0.04 * span, ymax + 0.06 * span); ax.set_xlabel("Rank in the top-100 list", fontsize=11); ax.set_ylabel("Score (absolute z)", fontsize=11)
    ax.set_title(title, fontsize=12.5, loc="left"); ax.set_facecolor("none")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9.5)
fig.suptitle(f"{ds}: top 100 TF–target pairs of each list (all TF × target Stage I pairs)", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/{ds}_top100_rank_score"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=130, facecolor="white"); print("saved", out)
