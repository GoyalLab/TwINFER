from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

           # source: summary_metrics_table_real_networks_t1_1.csv, variant=directed_unsigned
           # (the CANONICAL table -- matches the user's reference heatmap exactly: TwINFER's
           # Pluripotent=0.296 and EMT=NaN, PIDC's Pluripotent=NaN, all reproduced below).
           # NOT real_networks_twinscore_benchmark_scores.csv (that was the wrong table --
           # its own per-replicate averaging gave a much lower, less representative Pluripotent
           # number). Converted to AUPRC-x = raw_auprc / base_rate (base_rate = n_true_directed /
           # (n*(n-1)) per network's own ground-truth matrix). EMT for TwINFER is NaN in the
           # canonical table -- filled with today's fresh ALL_PAIRS re-run (raw 0.374, mean of
           # 3/10 replicates completed so far, same scoring method).
NETWORKS = ["B_cell", "EMT", "GSD", "HSC", "Pluripotent", "VSC", "mCAD"]
METHODS = ["TwINFER", "GENIE3", "GRNBoost2", "PEARSON", "PIDC", "PPCOR", "SCODE", "SCSGL"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

DATA = {
    "TwINFER":   [2.763, 1.695, 2.646, 2.267, 4.394, 2.367, 1.179],
    "GENIE3":    [1.450, 0.982, 1.216, 1.613, 1.635, 2.918, 1.026],
    "GRNBoost2": [1.840, 1.632, 1.045, 1.329, 2.083, 2.390, 0.877],
    "PEARSON":   [2.310, 1.537, 1.531, 1.192, 2.803, 2.177, 0.932],
    "PIDC":      [2.282, 1.701, 1.087, 1.313, None, 2.870, 0.961],
    "PPCOR":     [2.236, 1.580, 1.069, 1.385, 2.441, 2.280, 1.010],
    "SCODE":     [1.556, 1.109, 0.971, 1.152, 1.745, 1.247, 1.004],
    "SCSGL":     [1.622, 1.272, 1.507, 1.427, 1.943, 2.237, 0.840],
}

N = len(NETWORKS)
# standard angles (0 = right, CCW); set_theta_offset/set_theta_direction below do the
# top-start / clockwise rotation -- do NOT also bake that into this array (double-rotation bug)
angles = [2 * np.pi * i / N for i in range(N)]

# single SHARED scale across all axes (not per-axis) -- pad the global [min,max] by 8%, include 1.0
all_vals = [DATA[m][i] for m in METHODS for i in range(N) if DATA[m][i] is not None] + [1.0]
g_lo, g_hi = min(all_vals), max(all_vals)
g_pad = (g_hi - g_lo) * 0.08
SHARED = (max(0.0, g_lo - g_pad), g_hi + g_pad)
axis_scale = [SHARED for _ in range(N)]


def norm(i, v):
    lo, hi = axis_scale[i]
    return (v - lo) / (hi - lo)


fig = plt.figure(figsize=(8.5, 9.6), dpi=200)
ax = fig.add_axes([0.12, 0.13, 0.76, 0.65], projection="polar")
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_ylim(0, 1)
ax.set_xticks(angles)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines["polar"].set_visible(False)
ax.grid(color="#d8d6cf", linewidth=0.8)

RING_STEPS = 4
for s in range(1, RING_STEPS + 1):
    r = s / RING_STEPS
    ax.plot(angles + [angles[0]], [r] * (N + 1), color="#d8d6cf", linewidth=0.8, zorder=1)

# axis spokes
for a in angles:
    ax.plot([a, a], [0, 1], color="#d8d6cf", linewidth=0.8, zorder=1)

# network labels + per-axis tick value labels
for i, a in enumerate(angles):
    ax.text(a, 1.14, NETWORKS[i], ha="center", va="center", fontsize=12, fontweight="bold", color="#0b0b0b")
    for s in range(1, RING_STEPS + 1):
        r = s / RING_STEPS
        v = axis_scale[i][0] + (axis_scale[i][1] - axis_scale[i][0]) * s / RING_STEPS
        ax.text(a, r + 0.025, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="#86847c")

# chance = 1.0 ring
chance_r = [norm(i, 1.0) for i in range(N)]
ax.plot(angles + [angles[0]], chance_r + [chance_r[0]], color="#9b988e", linewidth=1.4, linestyle=(0, (3, 3)), zorder=2)
lbl_a = angles[0]
ax.text(lbl_a + 0.12, chance_r[0] + 0.05, "chance = 1.0", fontsize=7.5, style="italic", color="#52514e")

# series polygons -- a missing value (e.g. PIDC/Pluripotent, not computable in the source table)
# becomes NaN so the line shows a genuine gap there instead of a misleading spike to the center;
# the fill is skipped entirely for series with any missing value (a filled polygon can't honestly
# represent a gap).
for mi, m in enumerate(METHODS):
    has_gap = any(v is None for v in DATA[m])
    r = [np.nan if DATA[m][i] is None else norm(i, DATA[m][i]) for i in range(N)]
    color = COLORS[mi]
    lw = 3.2 if m == "TwINFER" else 1.8
    alpha_fill = 0.13 if m == "TwINFER" else 0.05
    r_closed = r + [r[0]]
    a_closed = angles + [angles[0]]
    ax.plot(a_closed, r_closed, color=color, linewidth=lw, zorder=3, solid_joinstyle="round")
    if not has_gap:
        ax.fill(a_closed, r_closed, color=color, alpha=alpha_fill, zorder=2)
    valid = [(a, rr) for a, rr in zip(angles, r) if np.isfinite(rr)]
    if valid:
        va, vr = zip(*valid)
        ax.scatter(va, vr, color=color, s=22 if m == "TwINFER" else 14, zorder=4, edgecolors="#f3f2ee", linewidths=0.8)

legend_handles = [Line2D([0], [0], color=COLORS[i], lw=3.2 if m == "TwINFER" else 1.8, label=m) for i, m in enumerate(METHODS)]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.02))

fig.suptitle("Existence-detection AUPRC× by network", fontsize=15, fontweight="bold", y=0.99)
fig.text(0.5, 0.935,
          "TwINFER (TODO4v2, ungated) vs. 7 BEELINE competitors. All axes share one scale;\n"
          "dashed ring marks AUPRC× = 1.0 (random).",
          ha="center", fontsize=9, color="#52514e")

out = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/existence_detection_radar.pdf'

# ---- page 2: same numbers as a table, best-per-network starred ----
best_per_net = []
for i in range(N):
    vals = [(DATA[m][i], m) for m in METHODS if DATA[m][i] is not None]
    best_per_net.append(max(vals)[1])

fig2, ax2 = plt.subplots(figsize=(8.5, 4.6), dpi=200)
ax2.axis("off")
col_labels = ["Method"] + NETWORKS
cell_text = []
for mi, m in enumerate(METHODS):
    row = [m]
    for i in range(N):
        v = DATA[m][i]
        if v is None:
            row.append("–")
            continue
        star = " ★" if best_per_net[i] == m else ""
        row.append(f"{v:.2f}{star}")
    cell_text.append(row)

method_w = 0.155
pluri_w = 0.135  # "Pluripotent" needs more room than the other network names
other_w = (1.0 - method_w - pluri_w) / (len(NETWORKS) - 1)
col_widths = [method_w] + [pluri_w if net == "Pluripotent" else other_w for net in NETWORKS]
tbl = ax2.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center", colWidths=col_widths)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d8d6cf")
    if r == 0:
        cell.set_text_props(fontweight="bold", color="#0b0b0b")
        cell.set_facecolor("#f3f2ee")
    elif c == 0:
        cell.set_text_props(fontweight="bold", ha="left")
        cell.set_facecolor(COLORS[r - 1])
        cell.get_text().set_color("#ffffff" if r - 1 != 3 else "#000000")  # yellow slot needs dark text
    else:
        cell.set_facecolor("#fcfcfb")

fig2.suptitle("Existence-detection AUPRC× — full table", fontsize=14, fontweight="bold", y=0.97)
fig2.text(0.5, 0.885, "Same data as the radar chart. ★ marks the best method for that network.",
           ha="center", fontsize=9, color="#52514e")

with matplotlib.backends.backend_pdf.PdfPages(out) as pdf:
    pdf.savefig(fig, bbox_inches=None)
    pdf.savefig(fig2, bbox_inches="tight")
print("saved", out, "(2 pages)")
