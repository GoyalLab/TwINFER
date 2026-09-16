"""
Plot every ground-truth network in input_data/mixed_network_sweep/ using the
same spread-layout style as generating_networks.ipynb (grn_plot_spread_v2):

    * filled black nodes on a fixed circular layout
    * blue "-|>" arrows for activation, red flat bars for repression
    * signed edge-weight colormap, vmin/vmax fixed to [-1, 1]

One figure per gene count (n6, n10). Panels are laid out with one row per
(cycle_len, density_level, autoreg_frac) config and one column per topology
replicate, read from topology_manifest.csv.

Self-loops (autoregulation) sit on the matrix diagonal. grn_plot_spread_v2
does not draw diagonal edges, so this script draws them itself as a small
loop on the outward side of each node (arrowhead = activation, flat bar =
repression), matching the activation/repression styling of the module.

Run:
    /home/gzu5140/.conda/envs/twinfer-code/bin/python \
        code/TwINFER/synthetic_network_analysis/plot_mixed_network_sweep.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from grn_plot_spread_v2 import plot_grn as plot_grn_spread
from grn_plot_spread_v2 import make_reds_blues_colormap
from matplotlib.colors import Normalize


NETWORK_DIR = "/home/gzu5140/TwINFER_KA/input_data/mixed_network_sweep"
OUTPUT_DIR = "/home/gzu5140/TwINFER_KA/network_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def circular_layout(n_nodes, radius=0.34):
    """g1 at top, going clockwise -- matches the notebook's hexagon feel."""
    angles = np.linspace(
        np.pi / 2.0,
        np.pi / 2.0 - 2.0 * np.pi,
        n_nodes,
        endpoint=False,
    )
    return {
        node: np.array([0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)])
        for node, a in enumerate(angles)
    }


HEX_POS = {
    0: np.array([0.28, 0.78]),  # g1 upper-left
    1: np.array([0.72, 0.78]),  # g2 upper-right
    2: np.array([0.86, 0.50]),  # g3 right
    3: np.array([0.72, 0.22]),  # g4 lower-right
    4: np.array([0.28, 0.22]),  # g5 lower-left
    5: np.array([0.14, 0.50]),  # g6 left
}


def load_matrix(path):
    matrix = np.loadtxt(path, delimiter=",")
    return np.asarray(matrix, dtype=float)  # diagonal (autoreg) kept


# module-matching geometry (grn_plot_spread_v2.plot_grn)
def _node_radius(n_nodes):
    return max(0.012, min(0.05, 0.15 / np.sqrt(n_nodes)))


def _line_width(n_nodes):
    side = max(6.0, min(20.0, np.sqrt(n_nodes) * 2.5))
    return 2.0 * (side / 8.0)


def draw_self_loops(ax, matrix, positions, n_nodes, cmap, norm):
    """Draw diagonal (self-regulatory) edges as small outward loops."""
    node_r = _node_radius(n_nodes)
    lw = _line_width(n_nodes)
    centroid = np.mean([positions[k] for k in range(n_nodes)], axis=0)
    loop_r = node_r * 0.95

    for node in range(n_nodes):
        weight = float(matrix[node, node])
        if weight == 0:
            continue
        color = cmap(norm(weight))
        center = np.asarray(positions[node], dtype=float)

        radial = center - centroid
        if np.linalg.norm(radial) < 1e-6:
            radial = np.array([0.0, 1.0])
        radial = radial / np.linalg.norm(radial)
        # node labels sit on the radial line; drop the loop ~55 deg off it so
        # the loop and the label do not overlap.
        theta = np.radians(55.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        outward = rot @ radial

        loop_center = center + outward * (node_r + loop_r * 0.85)
        gap_angle = np.arctan2(-outward[1], -outward[0])  # toward the node
        t = np.linspace(
            gap_angle + np.radians(40.0),
            gap_angle + np.radians(320.0),
            120,
        )
        xs = loop_center[0] + loop_r * np.cos(t)
        ys = loop_center[1] + loop_r * np.sin(t)

        ax.plot(xs, ys, color="white", lw=lw + 2.4, solid_capstyle="round", zorder=1.3)
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=2.0)

        end = np.array([xs[-1], ys[-1]])
        tangent = np.array([-np.sin(t[-1]), np.cos(t[-1])])
        tangent = tangent / np.linalg.norm(tangent)
        perp = np.array([-tangent[1], tangent[0]])

        if weight < 0:
            half = node_r * 0.55
            p1 = end - perp * half
            p2 = end + perp * half
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="white",
                    lw=lw * 1.5 + 2.4, solid_capstyle="butt", zorder=2.2)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                    lw=lw * 1.5, solid_capstyle="butt", zorder=2.6)
        else:
            head_l = node_r * 1.0
            head_w = node_r * 0.7
            tip = end + tangent * head_l * 0.5
            base = end - tangent * head_l * 0.5
            triangle = np.array([tip, base + perp * head_w / 2.0,
                                 base - perp * head_w / 2.0])
            ax.add_patch(plt.Polygon(triangle, closed=True, facecolor=color,
                                     edgecolor="white", lw=0.8, zorder=2.6))


def panel_title(row):
    autoreg = "autoreg" if row["n_self_loops"] > 0 else "no autoreg"
    return (
        f"cycle {int(row['cycle_len'])} / {int(row['offdiag_edge_count'])} edges / "
        f"{autoreg}\nrep {int(row['topology_replicate'])}"
    )


def make_figure(manifest, n_genes):
    sub = manifest[manifest["n_genes"] == n_genes].copy()
    sub = sub.sort_values(
        ["cycle_len", "density_level", "n_self_loops", "topology_replicate"]
    ).reset_index(drop=True)

    config_cols = ["cycle_len", "density_level", "n_self_loops"]
    configs = list(sub.groupby(config_cols, sort=False).groups.keys())
    reps = sorted(sub["topology_replicate"].unique())

    n_rows = len(configs)
    n_cols = len(reps)
    panel = 3.6

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel * n_cols, panel * n_rows),
        squeeze=False,
    )

    fixed_pos = HEX_POS if n_genes == 6 else circular_layout(n_genes)
    node_labels = [f"$g_{{{i + 1}}}$" for i in range(n_genes)]
    cmap = make_reds_blues_colormap(-1.0, 1.0)
    norm = Normalize(vmin=-1.0, vmax=1.0)

    for r, cfg in enumerate(configs):
        for c, rep in enumerate(reps):
            ax = axes[r][c]
            mask = np.ones(len(sub), dtype=bool)
            for col, val in zip(config_cols, cfg):
                mask &= sub[col] == val
            mask &= sub["topology_replicate"] == rep
            hits = sub[mask]
            if hits.empty:
                ax.axis("off")
                continue
            row = hits.iloc[0]
            matrix = load_matrix(os.path.join(NETWORK_DIR, row["filename"]))
            plot_grn_spread(
                matrix,
                node_labels=node_labels,
                title=panel_title(row),
                ax=ax,
                fixed_pos=fixed_pos,
                show_colorbar=True,
                font_size=11,
                vmin=-1,
                vmax=1,
                terminal_spread=1.1,
                terminal_angle_tolerance=1.10,
                reciprocal_separation=0.70,
                max_shift_fraction=1.0,
            )
            draw_self_loops(ax, matrix, fixed_pos, n_genes, cmap, norm)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.axis("off")

    fig.suptitle(
        f"Ground-Truth Networks: mixed_network_sweep (n = {n_genes} genes)",
        fontsize=19,
        fontweight="bold",
        y=0.997,
    )
    fig.subplots_adjust(
        left=0.02, right=0.985, bottom=0.02, top=0.955, wspace=0.28, hspace=0.30
    )

    stem = os.path.join(OUTPUT_DIR, f"mixed_network_sweep_ground_truth_n{n_genes}")
    fig.savefig(stem + ".png", dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.png / .pdf  ({n_rows}x{n_cols} panels)")


def main():
    manifest = pd.read_csv(os.path.join(NETWORK_DIR, "topology_manifest.csv"))
    for n_genes in sorted(manifest["n_genes"].unique()):
        make_figure(manifest, int(n_genes))


if __name__ == "__main__":
    main()
