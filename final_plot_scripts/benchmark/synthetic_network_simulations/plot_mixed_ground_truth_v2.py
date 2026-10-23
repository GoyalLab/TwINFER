"""Ground-truth mixed-sweep networks, drawn by this script rather than plot_grn, so arrowhead and bar size
are set in points and stay proportional to the panel. Same layout as plot_mixed_ground_truth.py: rows =
replicate networks 1-3, columns = the 16 mixed topologies in heatmap block order.

Edge rule (same as plot_grn): reciprocal pairs -- the higher-index -> lower-index edge is curved.
Activation (+1) = blue arrowhead; repression (-1) = red flat bar at the target. Colours are the
network_plots.py colormap endpoints (+1 blue, -1 red).

Input:  input_data/mixed_network_sweep/<stem>_rep<k>.txt
Output: final_plots/benchmark/synthetic_network_simulations/ground_truth_mixed_networks_v2.pdf

Date:   2026-10-04
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from matplotlib import font_manager as fm
ARIAL_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts"
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"]:
    fm.fontManager.addfont(os.path.join(ARIAL_DIR, f))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ["Arial"]
plt.rcParams['font.family'] = "sans-serif"

IN_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/input_data/mixed_network_sweep"
OUT = "/gpfs/projects/b1255/hzhang/TwINFER_KA/final_plots/benchmark/synthetic_network_simulations"
os.makedirs(OUT, exist_ok=True)

BLUE = "#08306b"   # network_plots.py colormap, weight +1
RED = "#67000d"    # network_plots.py colormap, weight -1
LW = 1.0           # edge width (points)
HEAD_PT = 4.5      # arrowhead size in points (mutation_scale)
BAR_HALF = 0.22    # repression bar half-length, in node radii
NODE_R = 0.055     # node radius in axes units (unit square)

BLOCKS = [
    ("6 genes, 6 edges", ["grn_n6_e6_c3_a0_pos50", "grn_n6_e6_c3_a3_pos50", "grn_n6_e6_c4_a0_pos50", "grn_n6_e6_c4_a3_pos50"]),
    ("6 genes, 12 edges", ["grn_n6_e12_c3_a0_pos50", "grn_n6_e12_c3_a3_pos50", "grn_n6_e12_c4_a0_pos50", "grn_n6_e12_c4_a3_pos50"]),
    ("10 genes, 10 edges", ["grn_n10_e10_c5_a0_pos50", "grn_n10_e10_c5_a5_pos50", "grn_n10_e10_c8_a0_pos50", "grn_n10_e10_c8_a5_pos50"]),
    ("10 genes, 20 edges", ["grn_n10_e20_c5_a0_pos50", "grn_n10_e20_c5_a5_pos50", "grn_n10_e20_c8_a0_pos50", "grn_n10_e20_c8_a5_pos50"]),
]
AUTOREG = {"a0": "No autoregulation", "a3": "Autoregulation", "a5": "Autoregulation"}


def header(stem):
    _, n, e, c, a, _ = stem.split("_")
    return f"Cycle {c[1:]}\n{AUTOREG[a]}"


def positions(n):
    """Circular layout, clockwise from upper-left, as in plot_grn."""
    angles = np.linspace(3 * np.pi / 4, 3 * np.pi / 4 - 2 * np.pi, n, endpoint=False)
    return np.column_stack([0.5 + 0.36 * np.cos(angles), 0.5 + 0.36 * np.sin(angles)])


def draw_network(ax, M):
    n = M.shape[0]
    P = positions(n)
    for i in range(n):
        for j in range(n):
            if i == j or M[i, j] == 0:
                continue
            curved = M[j, i] != 0 and i > j
            rad = 0.18 if curved else 0.0
            color = BLUE if M[i, j] > 0 else RED
            d = P[j] - P[i]
            u = d / np.linalg.norm(d)
            a = P[i] + u * NODE_R
            b = P[j] - u * NODE_R
            if M[i, j] > 0:
                ax.annotate("", xy=b, xytext=a, zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=LW, mutation_scale=HEAD_PT,
                                            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
            else:
                bar_c = P[j] - u * NODE_R * 1.0
                ax.annotate("", xy=bar_c, xytext=a, zorder=2,
                            arrowprops=dict(arrowstyle="-", color=color, lw=LW,
                                            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
                # flat bar perpendicular to the end of the edge
                t = (bar_c - a)
                ctrl = 0.5 * (a + bar_c) + rad * np.array([t[1], -t[0]])
                tang = bar_c - ctrl
                tang = tang / np.linalg.norm(tang)
                perp = np.array([-tang[1], tang[0]])
                half = NODE_R * BAR_HALF * 2
                p1, p2 = bar_c - perp * half, bar_c + perp * half
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=LW * 1.6, zorder=2.5, solid_capstyle="butt")
    for k in range(n):
        ax.add_patch(Circle(P[k], NODE_R, facecolor="#1a1a1a", edgecolor="white", lw=0.8, zorder=4))
        off = P[k] - np.array([0.5, 0.5])
        off = off / np.linalg.norm(off)
        ax.text(P[k][0] + off[0] * NODE_R * 2.2, P[k][1] + off[1] * NODE_R * 2.2, f"$g_{{{k + 1}}}$",
                ha="center", va="center", fontsize=8, zorder=5)


def main():
    stems = [s for _, group in BLOCKS for s in group]
    n_rows, n_cols = 3, len(stems)
    panel = 1.45
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel * n_cols + 1.0, panel * n_rows + 1.8))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.84, bottom=0.01, wspace=0.05, hspace=0.12)
    for j, stem in enumerate(stems):
        for i in range(n_rows):
            M = np.loadtxt(f"{IN_DIR}/{stem}_rep{i}.txt", delimiter=",")
            ax = axes[i, j]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.axis("off")
            draw_network(ax, M)
        axes[0, j].set_title(header(stem), fontsize=8, pad=4)
    for i in range(n_rows):
        axes[i, 0].text(-0.12, 0.5, f"Network {i + 1}", transform=axes[i, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=9, fontweight="bold")
    col = 0
    for block, group in BLOCKS:
        left = axes[0, col].get_position().x0
        right = axes[0, col + len(group) - 1].get_position().x1
        fig.text((left + right) / 2, 0.955, block, ha="center", va="center", fontsize=10, fontweight="bold")
        fig.add_artist(plt.Line2D([left, right], [0.94, 0.94], color="black", lw=0.8, transform=fig.transFigure))
        col += len(group)
    fig.savefig(f"{OUT}/ground_truth_mixed_networks_v2.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f"{OUT}/ground_truth_mixed_networks_v2.pdf")


if __name__ == "__main__":
    main()
