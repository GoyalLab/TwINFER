#!/usr/bin/env python3
"""Four candidate designs (one contact sheet) for showing, for FM01 and FM06 in ONE plot, which Gavish et al. 2023 meta-program groups the targets of the top-1000 unshared and shared lists belong to (descriptive, no uncertainty).
Counts: ChIP-Atlas candidate edges, Stage I, no gene filter; AEBP1, CENPA, HMGA1, YBX1 and non-Lambert regulators excluded; 'target in group' as in panel b of yscher's figure (a target in several groups counts in each),
except option 4, which needs every pair in ONE class and uses the first matching group of the order Cell cycle > Skin pigmentation > Mesenchymal > Stress/hypoxia > Secreted > MYC > Astrocyte/NPC/OPC > other program > none.
Output: analysis_data/fm_meta_programs_comparison/program_group_plot_options.png   usage: program_group_plot_options.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; N = 1000
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
X = pd.read_excel("/home/gzu5140/TwINFER_KA/resources/41586_2023_6130_MOESM6_ESM.xlsx", sheet_name="Cancer MPs", header=None)
mp = {int(str(X.iloc[0, j]).split()[0][2:]): set(str(g) for g in X.iloc[1:, j].dropna()) for j in range(X.shape[1])}; anymp = set().union(*mp.values())
G = [("Cell cycle", [1, 2, 3, 4], "#d62728"), ("Skin pigmentation", [32], "#2ca02c"), ("Mesenchymal", [12, 13, 14, 15, 16], "#1f77b4"), ("Stress, hypoxia", [5, 6, 7], "#ff7f0e"), ("Secreted", [22, 23], "#8c564b"), ("MYC", [20], "#9467bd"), ("Astrocyte, NPC, OPC", [25, 26, 27, 28, 29], "#17becf")]
GS = {n: set().union(*[mp[m] for m in ms]) for n, ms, _ in G}; COL = {n: c for n, _, c in G}; COL["other meta-program"] = "#7f7f7f"; COL["no meta-program"] = "#e3e3e3"
def excl(g):
    for n, _, _ in G:
        if g in GS[n]: return n
    return "other meta-program" if g in anymp else "no meta-program"
cnt, exc = {}, {}
for ds in ("FM01", "FM06"):
    P = pd.read_csv(f"{A}/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]; P = P[~(P.TF.isin(four) | ~P.TF.isin(lam))]
    for lab, k in (("Unshared", "z_U"), ("Shared", "h")):
        d = P.assign(s=P[k].abs()).nlargest(N, "s"); cnt[(ds, lab)] = {n: int(d.gene.isin(GS[n]).sum()) for n, _, _ in G}; exc[(ds, lab)] = d.gene.map(excl).value_counts().to_dict()
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
CU, CS = "#6a54d3", "#1fb07b"; names = [n for n, _, _ in G]; pct = lambda ds, lab, n: 100 * cnt[(ds, lab)][n] / N
fig, axs = plt.subplots(2, 2, figsize=(17.5, 12.6)); fig.patch.set_alpha(0)
# ---- option 1: paired strip, both datasets in one panel (colour = group, same colour in both datasets)
ax = axs[0, 0]; xp = {("FM01", "Unshared"): 0, ("FM01", "Shared"): 1, ("FM06", "Unshared"): 2.7, ("FM06", "Shared"): 3.7}
for n in names:
    for ds in ("FM01", "FM06"): ax.plot([xp[(ds, "Unshared")], xp[(ds, "Shared")]], [pct(ds, "Unshared", n), pct(ds, "Shared", n)], "-o", color=COL[n], lw=2.2, ms=6, mec="white", mew=0.7)
    ax.text(-0.12, pct("FM01", "Unshared", n), n, ha="right", va="center", fontsize=9, color=COL[n])
ax.set_xticks(list(xp.values())); ax.set_xticklabels(["Unshared", "Shared", "Unshared", "Shared"], fontsize=10); ax.set_xlim(-1.5, 3.95); ax.text(0.5, -0.115, "FM01", transform=ax.get_xaxis_transform(), ha="center", fontsize=12, fontweight="bold"); ax.text(3.2, -0.115, "FM06", transform=ax.get_xaxis_transform(), ha="center", fontsize=12, fontweight="bold")
ax.axvline(1.85, color="#ddd", lw=1); ax.set_ylabel("% of the top-1000 pairs with a target in the group", fontsize=10); ax.set_title("Option 1: paired strip, both datasets in one panel", loc="left", fontsize=12, fontweight="bold")
# ---- option 2: dot matrix (circle area = %)
ax = axs[0, 1]; cols = [("FM01", "Unshared"), ("FM01", "Shared"), ("FM06", "Unshared"), ("FM06", "Shared")]
for j, c in enumerate(cols):
    for i, n in enumerate(names):
        v = pct(*c, n); ax.scatter(j + (0.35 if j >= 2 else 0), i, s=v * 62, color=CU if c[1] == "Unshared" else CS, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3); ax.text(j + (0.35 if j >= 2 else 0), i, f"{v:.1f}", ha="center", va="center", fontsize=8.2, color="#111", zorder=5)
ax.set_xticks([0, 1, 2.35, 3.35]); ax.set_xticklabels(["Unshared", "Shared", "Unshared", "Shared"], fontsize=10); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10); ax.invert_yaxis(); ax.set_xlim(-0.6, 3.95); ax.set_ylim(len(names) - 0.4, -0.7)
ax.text(0.5, -0.12, "FM01", transform=ax.get_xaxis_transform(), ha="center", fontsize=12, fontweight="bold"); ax.text(2.85, -0.12, "FM06", transform=ax.get_xaxis_transform(), ha="center", fontsize=12, fontweight="bold"); ax.axvline(1.75, color="#ddd", lw=1)
ax.set_title("Option 2: dot matrix (circle area = % of the list; number = %)", loc="left", fontsize=12, fontweight="bold"); ax.tick_params(length=0)
# ---- option 3: dumbbell, FM01 and FM06 sub-rows per group
ax = axs[1, 0]; y = 0; ticks = []
for n in names:
    for k, ds in enumerate(("FM01", "FM06")):
        u, s = pct(ds, "Unshared", n), pct(ds, "Shared", n); yy = y + k * 0.42; ax.plot([u, s], [yy, yy], "-", color=CS if s > u + 0.5 else (CU if u > s + 0.5 else "#aaa"), lw=3, zorder=2); ax.scatter([u], [yy], s=55, color=CU, edgecolor="white", zorder=3); ax.scatter([s], [yy], s=55, color=CS, edgecolor="white", zorder=3)
        ax.text(-0.4, yy, ds, ha="right", va="center", fontsize=8, color="#666")
    ticks.append(y + 0.21); y += 1.15
ax.set_yticks(ticks); ax.set_yticklabels(names, fontsize=10); ax.invert_yaxis(); ax.set_xlabel("% of the top-1000 pairs with a target in the group", fontsize=10); ax.set_xlim(-0.5, 22); ax.tick_params(axis="y", length=0)
ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=CU, markersize=8, label="unshared list"), Line2D([0], [0], marker="o", color="none", markerfacecolor=CS, markersize=8, label="shared list")], frameon=False, fontsize=9.5, loc="lower right")
ax.set_title("Option 3: dumbbell (line joins unshared and shared; FM01 above FM06)", loc="left", fontsize=12, fontweight="bold")
# ---- option 4: composition of each list (stacked, sums to 100%)
ax = axs[1, 1]; order = names + ["other meta-program", "no meta-program"]; xs = [0, 1, 2.7, 3.7]
for x, c in zip(xs, cols):
    bot = 0
    for n in order:
        v = 100 * exc[c].get(n, 0) / N; ax.bar(x, v, bottom=bot, width=0.8, color=COL[n], edgecolor="white", linewidth=0.6)
        if v >= 3.5 and n != "no meta-program": ax.text(x, bot + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8.2, color="white" if n not in ("other meta-program",) else "white")
        bot += v
ax.set_xticks(xs); ax.set_xticklabels(["Unshared", "Shared", "Unshared", "Shared"], fontsize=10); ax.set_ylim(0, 100); ax.set_ylabel("% of the top-1000 pairs of the list", fontsize=10); ax.text(0.5, -0.115, "FM01", transform=ax.get_xaxis_transform(), ha="center", fontsize=12, fontweight="bold"); ax.text(3.2, -0.115, "FM06", transform=ax.get_xaxis_transform(), ha="center", fontsize=12, fontweight="bold")
ax.legend(handles=[Line2D([0], [0], color=COL[n], lw=6, label=n) for n in order[::-1]], frameon=False, fontsize=8.8, loc="center left", bbox_to_anchor=(1.0, 0.5)); ax.set_title("Option 4: composition of each list (each pair counted once)", loc="left", fontsize=12, fontweight="bold")
for ax in axs.flat:
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.set_facecolor("none")
fig.suptitle("Options for one plot of FM01 and FM06: meta-program groups of the target genes in the top-1000 unshared and shared lists (ChIP-Atlas edges)", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=3.2, w_pad=3.5)
fig.savefig(f"{A}/fm_meta_programs_comparison/program_group_plot_options.png", dpi=95, facecolor="white"); print("saved")
