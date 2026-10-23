#!/usr/bin/env python3
"""Dot-and-whisker plot: for each Gavish et al. 2023 meta-program group (grouping of panel b of yscher's figure), the UNSHARED SHARE of the pairs whose target is in the group =
n_unshared / (n_unshared + n_shared), counted over the pairs of the top-1000 unshared list and the top-1000 shared list (ChIP-Atlas candidate edges, Stage I, no gene filter; AEBP1, CENPA, HMGA1, YBX1 and
non-Lambert regulators excluded; a pair in both lists counts on both sides), with an exact (Clopper-Pearson) 95 % interval, for FM01 and FM06.  50 % = no preference; right = unshared, left = shared.
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/fm01_fm06_program_group_unshared_share.{pdf,png} and a csv with the counts.   usage: program_group_unshared_share_plot.py"""
import numpy as np, pandas as pd
from scipy.stats import binomtest
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; N = 1000
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
X = pd.read_excel("/home/gzu5140/TwINFER_KA/resources/41586_2023_6130_MOESM6_ESM.xlsx", sheet_name="Cancer MPs", header=None)
mp = {int(str(X.iloc[0, j]).split()[0][2:]): set(str(g) for g in X.iloc[1:, j].dropna()) for j in range(X.shape[1])}
G = [("Cell cycle", [1, 2, 3, 4]), ("MYC", [20]), ("Secreted", [22, 23]), ("Stress, hypoxia", [5, 6, 7]), ("Astrocyte, NPC, OPC", [25, 26, 27, 28, 29]), ("Mesenchymal", [12, 13, 14, 15, 16]), ("Skin pigmentation", [32])]
rows = []
for ds in ("FM01", "FM06"):
    P = pd.read_csv(f"{A}/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]; P = P[~(P.TF.isin(four) | ~P.TF.isin(lam))].reset_index(drop=True)
    u = P.assign(s=P.z_U.abs()).nlargest(N, "s"); s = P.assign(s=P.h.abs()).nlargest(N, "s")
    for name, ms in G:
        gs = set().union(*[mp[m] for m in ms]); nu, ns = int(u.gene.isin(gs).sum()), int(s.gene.isin(gs).sum()); t = nu + ns; ci = binomtest(nu, t, 0.5).proportion_ci(0.95, method="exact")
        rows.append(dict(dataset=ds, group=name, n_unshared=nu, n_shared=ns, unshared_share=nu / t, ci_low=ci.low, ci_high=ci.high))
T = pd.DataFrame(rows); T.to_csv(f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_unshared_share.csv", index=False)
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
COL = {"FM01": "#6a54d3", "FM06": "#1fb07b"}; order = [g for g, _ in G]; ypos = {g: i for i, g in enumerate(order)}
fig, ax = plt.subplots(figsize=(8.6, 5.4)); fig.patch.set_alpha(0); ax.set_facecolor("none")
ax.axvspan(0, 50, color="#f3f3f3", zorder=0, lw=0); ax.axvline(50, color="#777", lw=1, ls="--", zorder=1)
for ds, off in (("FM01", 0.13), ("FM06", -0.13)):
    d = T[T.dataset == ds]
    for r in d.itertuples():
        y = ypos[r.group] + off; ax.plot([100 * r.ci_low, 100 * r.ci_high], [y, y], "-", color=COL[ds], lw=2.2, zorder=2, solid_capstyle="round"); ax.plot(100 * r.unshared_share, y, "o", ms=8.5, color=COL[ds], markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(101.5, y, f"{r.n_unshared} / {r.n_shared}", fontsize=8, color=COL[ds], va="center", ha="left")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=11); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_ylim(len(order) - 0.45, -0.6)
ax.set_xlabel("Unshared share of the pairs whose target is in the group:  n unshared / (n unshared + n shared)", fontsize=10.5); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
ax.text(0, -0.9, "← leans shared", fontsize=10, color="#555", ha="left", va="bottom"); ax.text(100, -0.9, "leans unshared →", fontsize=10, color="#555", ha="right", va="bottom"); ax.text(101.5, -0.9, "n unsh. / shared", fontsize=8, color="#555", ha="left", va="bottom")
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.tick_params(axis="y", length=0)
for ds, c in COL.items(): ax.plot([], [], "o", color=c, ms=8, label=f"{ds}")
ax.legend(loc="lower left", frameon=False, fontsize=10, bbox_to_anchor=(0.0, -0.29), ncol=2)
fig.suptitle("Which list do the targets of each meta-program group fall in?", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=0.995)
fig.text(0.01, 0.935, "top-1000 shared and unshared lists, ChIP-Atlas edges; dots = unshared share, bars = exact 95% interval; dashed line = 50% (no preference)", fontsize=8.3, color="#666", ha="left")
fig.tight_layout(rect=[0, 0.02, 0.93, 0.92])
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_unshared_share"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=140, facecolor="white"); print("saved", out)
