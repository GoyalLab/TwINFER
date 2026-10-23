#!/usr/bin/env python3
"""Back-to-back bar chart (as panel b of yscher's figure): one row per Gavish et al. 2023 meta-program group; left = number of the top-1000 UNSHARED pairs whose target is in the group, right = the same for the top-1000
SHARED pairs; two bars per side and group, FM01 (lighter) above FM06 (darker).  ChIP-Atlas candidate edges, Stage I, no gene filter; AEBP1, CENPA, HMGA1, YBX1 and non-Lambert regulators excluded; a target in several groups
counts in each.  Counts from fm01_fm06_program_group_unshared_share.csv.  Rows ordered from unshared-leaning (top) to shared-leaning (bottom).
Schaffer naive = melanoma ChIP-Atlas edges (FM01 epidermis list), BH recomputed (schaffer_naive_program_group_counts.csv, mode chip_fm01_epidermis). Watermelon naive (T47D, breast) is the last bar of each group: ChIP-Atlas BREAST-class edges (analysis_data/chipatlas_hg38_watermelon_breast_class, hg38, +-5 kb, score >= 100, measured genes), BH recomputed over those edges (counts in watermelon_naive_program_group_counts.csv, mode chip_breast).
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/fm01_fm06_schaffer_watermelon_program_group_butterfly.{pdf,png}.   usage: program_group_butterfly_plot.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"
T = pd.read_csv(f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_unshared_share.csv")
W = pd.read_csv(f"{A}/fm_meta_programs_comparison/watermelon_naive_program_group_counts.csv"); W = W[W["mode"] == "chip_breast"].assign(dataset="WM naive")[["dataset", "group", "n_unshared", "n_shared"]]; S = pd.read_csv(f"{A}/fm_meta_programs_comparison/schaffer_naive_program_group_counts.csv"); S = S[S["mode"] == "chip_fm01_epidermis"].assign(dataset="Schaffer")[["dataset", "group", "n_unshared", "n_shared"]]   # Schaffer 2026 naive melanoma: the same FM01 ChIP-Atlas (epidermis) edges as FM01/FM06, BH over those edges
T = pd.concat([T[["dataset", "group", "n_unshared", "n_shared"]], S, W], ignore_index=True)
order = ["Cell cycle", "MYC", "Secreted", "Stress, hypoxia", "Astrocyte, NPC, OPC", "Skin pigmentation", "Mesenchymal"]
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
C = {("FM01", "unshared"): "#c9c0f2", ("FM06", "unshared"): "#9a8bea", ("Schaffer", "unshared"): "#6a54d3", ("WM naive", "unshared"): "#33258a", ("FM01", "shared"): "#b5ead8", ("FM06", "shared"): "#7fd6b8", ("Schaffer", "shared"): "#1fb07b", ("WM naive", "shared"): "#0b6b4a"}; GAP = 62; H = 0.19; DS = ["FM01", "FM06", "Schaffer", "WM naive"]
fig, ax = plt.subplots(figsize=(11.4, 8.4)); fig.patch.set_alpha(0); ax.set_facecolor("none")
for i, g in enumerate(order):
    for k, (ds, off) in enumerate(zip(DS, (-0.30, -0.10, 0.10, 0.30))):
        r = T[(T.dataset == ds) & (T.group == g)].iloc[0]; nu, ns = int(r.n_unshared), int(r.n_shared); y = i + off
        ax.barh(y, nu, left=-GAP - nu, height=H, color=C[(ds, "unshared")], edgecolor="none"); ax.barh(y, ns, left=GAP, height=H, color=C[(ds, "shared")], edgecolor="none")
        ax.text(-GAP - nu - 4, y, str(nu), ha="right", va="center", fontsize=9, color="#333"); ax.text(GAP + ns + 4, y, str(ns), ha="left", va="center", fontsize=9, color="#333")
    ax.text(0, i, g, ha="center", va="center", fontsize=11, color="#111")
for v in (0, 100, 200):
    ax.plot([GAP + v] * 2, [-0.6, len(order) - 0.4], color="#ececec", lw=0.8, zorder=0); ax.plot([-GAP - v] * 2, [-0.6, len(order) - 0.4], color="#ececec", lw=0.8, zorder=0)
    ax.text(GAP + v, len(order) - 0.28, str(v), ha="center", va="top", fontsize=8.5, color="#888"); ax.text(-GAP - v, len(order) - 0.28, str(v), ha="center", va="top", fontsize=8.5, color="#888")
ax.text(-GAP - 120, -0.95, "Unshared top 1000", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#5a46bd"); ax.text(GAP + 120, -0.95, "Shared top 1000", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#17986b")
ax.set_xlim(-GAP - 290, GAP + 290); ax.set_ylim(len(order) - 0.1, -1.0); ax.axis("off")
ax.text(0, len(order) + 0.55, "number of pairs out of 1000 whose target is in the group", ha="center", va="top", fontsize=9, color="#777")
fig.legend(handles=[Patch(color=C[(d, l)], label=f"{d} {l}") for l in ("unshared", "shared") for d in DS], loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Which meta-program group do the targets of the top-1000 pairs belong to?", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=0.985)
fig.text(0.01, 0.925, "ChIP-Atlas edges. Bars per group, top to bottom: FM01, FM06, Schaffer naive (melanoma edges), Watermelon naive (breast edges, T47D); a target counts in every group it belongs to", fontsize=8.3, color="#666")
fig.tight_layout(rect=[0, 0.05, 1, 0.91])
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm01_fm06_schaffer_watermelon_program_group_butterfly"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=140, facecolor="white"); print("saved", out)
