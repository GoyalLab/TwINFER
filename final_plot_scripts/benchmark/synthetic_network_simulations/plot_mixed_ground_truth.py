"""Ground-truth mixed-sweep networks, drawn with the plot_grn style used for the network-sweep reference figure
(blue activation arrows, red repression bars, reciprocal edges curved). Rows = replicate networks 1-3,
columns = the 16 mixed topologies in heatmap block order (6 genes/6 edges, 6/12, 10/10, 10/20), each with
Cycle header and No autoregulation / Autoregulation sub-header.

Input:  input_data/mixed_network_sweep/<stem>_rep<k>.txt  (signed matrix: +1 activation, -1 repression)
Output: final_plots/benchmark/synthetic_network_simulations/ground_truth_mixed_networks.pdf

Date:   2026-10-04
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from twinfer.plotting.network_plots import plot_grn

from matplotlib import font_manager as fm
ARIAL_DIR = "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts"
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"]:
    fm.fontManager.addfont(os.path.join(ARIAL_DIR, f))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ["Arial"]
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['mathtext.fontset'] = "cm"

IN_DIR = "/projects/b1255/hzhang/TwINFER_KA/input_data/mixed_network_sweep"
OUT = "/projects/b1255/hzhang/TwINFER_KA/final_plots/benchmark/synthetic_network_simulations"
os.makedirs(OUT, exist_ok=True)

# (block heading, [stems in column order]) -- stems are the file names without _rep<k>.txt
BLOCKS = [
    ("6 genes, 6 edges", ["grn_n6_e6_c3_a0_pos50", "grn_n6_e6_c3_a3_pos50", "grn_n6_e6_c4_a0_pos50", "grn_n6_e6_c4_a3_pos50"]),
    ("6 genes, 12 edges", ["grn_n6_e12_c3_a0_pos50", "grn_n6_e12_c3_a3_pos50", "grn_n6_e12_c4_a0_pos50", "grn_n6_e12_c4_a3_pos50"]),
    ("10 genes, 10 edges", ["grn_n10_e10_c5_a0_pos50", "grn_n10_e10_c5_a5_pos50", "grn_n10_e10_c8_a0_pos50", "grn_n10_e10_c8_a5_pos50"]),
    ("10 genes, 20 edges", ["grn_n10_e20_c5_a0_pos50", "grn_n10_e20_c5_a5_pos50", "grn_n10_e20_c8_a0_pos50", "grn_n10_e20_c8_a5_pos50"]),
]
# plot_grn scales arrowheads and lines by min(figsize)/8; a small figsize gives heads that fit the small panels
FIG_SCALE = 4.0
AUTOREG = {"a0": "No autoregulation", "a3": "Autoregulation", "a5": "Autoregulation"}


def header(stem):
    _, n, e, c, a, _ = stem.split("_")
    return f"Cycle {c[1:]}\n{AUTOREG[a]}"


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
            plot_grn(M, ax=ax, figsize=(FIG_SCALE, FIG_SCALE), show_colorbar=False, font_size=6, label_font_size=7)
            ax.set_title("")
        axes[0, j].set_title(header(stem), fontsize=8, pad=4)
    for i in range(n_rows):
        axes[i, 0].text(-0.12, 0.5, f"Network {i + 1}", transform=axes[i, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=9, fontweight="bold")
    # block headings and rules across the top
    col = 0
    for block, group in BLOCKS:
        left = axes[0, col].get_position().x0
        right = axes[0, col + len(group) - 1].get_position().x1
        fig.text((left + right) / 2, 0.955, block, ha="center", va="center", fontsize=10, fontweight="bold")
        fig.add_artist(plt.Line2D([left, right], [0.94, 0.94], color="black", lw=0.8, transform=fig.transFigure))
        col += len(group)
    fig.savefig(f"{OUT}/ground_truth_mixed_networks.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f"{OUT}/ground_truth_mixed_networks.pdf")


if __name__ == "__main__":
    main()
