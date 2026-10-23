"""Existence-detection AUPRC-x radar plot and table, real-network simulations.

Date:   2026-10-04
Data:   summary_metrics_table_real_networks_t1_1.csv (variant = directed_unsigned, canonical table
        matching the reference heatmap), raw AUPRC per method x network, converted to
        AUPRC-x = raw AUPRC / base_rate, base_rate = n_true_directed / (n*(n-1)) from each network's
        ground-truth matrix in input_data/real_world_networks.
        TwINFER (TODO4v2) EMT is missing in that table; filled with the mean raw AUPRC (0.3756, 10/10 replicates) of the
        10/10 ALL_PAIRS replicates (job 7768362, completed 2026-10-04; same scoring method). PIDC Pluripotent is
        missing in the source table and is drawn as a gap.
Output: PDF (vector, transparent), written to analysis_data (not to the repo).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib import font_manager as fm
ARIAL_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts"
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"]:
    fm.fontManager.addfont(os.path.join(ARIAL_DIR, f))

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ["Arial"]
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['svg.fonttype'] = "none"
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

OUT_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/final_plots/benchmark/real_network_simulations"
os.makedirs(OUT_DIR, exist_ok=True)

NETWORKS = ["B_cell", "EMT", "GSD", "HSC", "Pluripotent", "VSC", "mCAD"]
METHODS = ["TwINFER", "GENIE3", "GRNBoost2", "PEARSON", "PIDC", "PPCOR", "SCODE", "SCSGL"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# AUPRC-x per method (rows) x network (cols), read from the summary CSV (no typed numbers).
# AUPRC-x = raw AUPRC / base rate, base rate = n_true_directed / (n*(n-1)) from each network's ground-truth matrix.
SUMMARY_CSV = "/home/gzu5140/TwINFER_KA/clean_data/benchmarks/network_benchmarks/score_results/summary_metrics_table_real_networks_t1_1_20261006.csv"
RW_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/input_data/real_world_networks"
DATASET_OF = {"B_cell": "B_cell_activation", "EMT": "EMT", "GSD": "GSD", "HSC": "HSC", "Pluripotent": "Pluripotent",
              "VSC": "VSC", "mCAD": "mCAD"}
MATRIX_OF = {"B_cell": "B_cell.txt", "EMT": "EMT.txt", "GSD": "GSD.txt", "HSC": "HSC.txt", "Pluripotent": "Pluripotent.txt",
             "VSC": "VSC.txt", "mCAD": "mCAD.txt"}
METHOD_OF = {"TwINFER": "TODO4v2", "GENIE3": "GENIE3", "GRNBoost2": "GRNBOOST2", "PEARSON": "PEARSON", "PIDC": "PIDC",
             "PPCOR": "PPCOR", "SCODE": "SCODE", "SCSGL": "SCSGL"}


def base_rate(matrix_file):
    Mx = np.loadtxt(os.path.join(RW_DIR, matrix_file), delimiter=",")
    n = Mx.shape[0]
    return np.count_nonzero(Mx) / (n * (n - 1))


_summary = pd.read_csv(SUMMARY_CSV)
_summary = _summary[_summary.variant == "directed_unsigned"]
_base = {net: base_rate(MATRIX_OF[net]) for net in NETWORKS}
DATA = {}
for m, src in METHOD_OF.items():
    row = []
    for net in NETWORKS:
        r = _summary[(_summary.method == src) & (_summary.dataset_id == DATASET_OF[net])]
        row.append(float(r.auprc.iloc[0]) / _base[net] if len(r) else None)
    DATA[m] = row

N = len(NETWORKS)
angles = [2 * np.pi * i / N for i in range(N)]

all_vals = [v for m in METHODS for v in DATA[m] if v is not None] + [1.0]
g_lo, g_hi = min(all_vals), max(all_vals)
pad = (g_hi - g_lo) * 0.08
SCALE_LO, SCALE_HI = max(0.0, g_lo - pad), g_hi + pad


def norm(v):
    return (v - SCALE_LO) / (SCALE_HI - SCALE_LO)


# ---------------- radar ----------------
fig = plt.figure(figsize=(6, 7.2))
ax = fig.add_axes([0.14, 0.2, 0.72, 0.72], projection="polar")
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_ylim(0, 1)
ax.set_xticks(angles)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines["polar"].set_visible(False)
ax.grid(color="#d0d0d0", linewidth=0.8)

for s in range(1, 5):
    ax.plot(angles + [angles[0]], [s / 4] * (N + 1), color="#d0d0d0", linewidth=0.8, zorder=1)
for a in angles:
    ax.plot([a, a], [0, 1], color="#d0d0d0", linewidth=0.8, zorder=1)

for i, a in enumerate(angles):
    ax.text(a, 1.26, NETWORKS[i].replace("_", " "), ha="center", va="center", fontsize=12, color="black")
    for s in range(1, 5):
        v = SCALE_LO + (SCALE_HI - SCALE_LO) * s / 4
        ax.text(a, s / 4 + 0.03, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#444444", zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))

chance = [norm(1.0)] * N
ax.plot(angles + [angles[0]], chance + [chance[0]], color="gray", linewidth=1.4, linestyle="--", zorder=2)

for mi, m in enumerate(METHODS):
    vals = DATA[m]
    r = [np.nan if v is None else norm(v) for v in vals]
    lw = 3.0 if m == "TwINFER" else 1.5
    ax.plot(angles + [angles[0]], r + [r[0]], color=COLORS[mi], linewidth=lw, zorder=3)
    if all(v is not None for v in vals):
        ax.fill(angles + [angles[0]], r + [r[0]], color=COLORS[mi], alpha=0.10 if m == "TwINFER" else 0.04, zorder=2)
    valid = [(a, rr) for a, rr in zip(angles, r) if np.isfinite(rr)]
    if valid:
        va, vr = zip(*valid)
        ax.scatter(va, vr, color=COLORS[mi], s=18, zorder=4, edgecolors="white", linewidths=0.6)

handles = [Line2D([0], [0], color=COLORS[i], lw=3.0 if m == "TwINFER" else 1.5, label=m) for i, m in enumerate(METHODS)]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))

fig.savefig(os.path.join(OUT_DIR, "auprc_x_radar_real_networks.pdf"), format="pdf",
            bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
plt.close(fig)

# ---------------- table ----------------
best = []
for i in range(N):
    cand = [(DATA[m][i], m) for m in METHODS if DATA[m][i] is not None]
    best.append(max(cand)[1])

cell_text = []
for m in METHODS:
    row = []
    for i in range(N):
        v = DATA[m][i]
        if v is None:
            row.append("–")
        else:
            row.append(f"{v:.2f}")
    cell_text.append(row)

fig2, ax2 = plt.subplots(figsize=(8, 3.6))
ax2.axis("off")
tbl = ax2.table(cellText=cell_text, rowLabels=METHODS, colLabels=[n.replace("_", " ") for n in NETWORKS],
                cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cfcfcf")
    if r == 0 or c == -1:
        cell.set_text_props(fontweight="bold")
for mi, m in enumerate(METHODS):
    for i in range(N):
        if DATA[m][i] is not None and best[i] == m:
            tbl[(mi + 1, i)].set_text_props(fontweight="bold")
fig2.savefig(os.path.join(OUT_DIR, "auprc_x_table_real_networks.pdf"), format="pdf",
             bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
plt.close(fig2)

print("wrote", os.path.join(OUT_DIR, "auprc_x_radar_real_networks.pdf"))
print("wrote", os.path.join(OUT_DIR, "auprc_x_table_real_networks.pdf"))
