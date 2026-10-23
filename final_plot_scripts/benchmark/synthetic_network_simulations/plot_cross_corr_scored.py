"""Directed-only cross-correlation versions of the AUPRC× radars and heatmaps.
TwINFER = directional cross-correlation ranking (rho_cross_xy for x->y, rho_cross_yx for y->x), ranked separately.
Competitors and their values are unchanged from the existing figures:
  real networks: corrected summary (summary_metrics_table_real_networks_t1_1_20261006.csv, canonical ground truth)
  sweeps:        existing auprc_x_<family>.csv and heatmap_data_<family>.csv (competitor rows)
Signed and undirected variants are not produced (the cross-correlation score is directed-unsigned only).
Outputs: cross_corr_scored_*.pdf in the same folders as the TODO4v2 figures.
Date: 2026-10-06
"""
import os
import re
import sys
from pathlib import Path

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
plt.rcParams['font.sans-serif'] = ["Arial"]
plt.rcParams['font.family'] = "sans-serif"

ROOT = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
RW = ROOT / "input_data" / "real_world_networks"
SYN = ROOT / "analysis_data" / "synthetic_network_simulations"
REAL_OUT = ROOT / "final_plots" / "benchmark" / "real_network_simulations"
SYN_OUT = ROOT / "final_plots" / "benchmark" / "synthetic_network_simulations"
SUMMARY = Path("/home/gzu5140/TwINFER_KA/clean_data/benchmarks/network_benchmarks/score_results/summary_metrics_table_real_networks_t1_1_20261006.csv")
HEATDATA = Path("/home/gzu5140/TwINFER_KA/clean_data/benchmarks/network_benchmarks/score_results")

METHODS = ["TwINFER", "GENIE3", "GRNBoost2", "PEARSON", "PIDC", "PPCOR", "SCODE", "SCSGL"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SUMMARY_NAME = {"TwINFER": "TODO4v2", "GENIE3": "GENIE3", "GRNBoost2": "GRNBOOST2", "PEARSON": "PEARSON",
                "PIDC": "PIDC", "PPCOR": "PPCOR", "SCODE": "SCODE", "SCSGL": "SCSGL"}
PRETTY = {"TwINFER": "TwINFER (cross-correlation)", "GRNBoost2": "GRNBoost2"}


def draw_radar(names, data, scale_lo, scale_hi, title_note, out_path):
    N = len(names)
    angles = [2 * np.pi * i / N for i in range(N)]

    def norm(v):
        return (v - scale_lo) / (scale_hi - scale_lo)

    fig = plt.figure(figsize=(6.2, 7.2))
    ax = fig.add_axes([0.14, 0.2, 0.72, 0.72], projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    for s in range(1, 5):
        ax.plot(angles + [angles[0]], [s / 4] * (N + 1), color="#d0d0d0", linewidth=0.8, zorder=1)
    for a in angles:
        ax.plot([a, a], [0, 1], color="#d0d0d0", linewidth=0.8, zorder=1)
    for i, a in enumerate(angles):
        ax.text(a, 1.26, names[i], ha="center", va="center", fontsize=10, color="black")
        for s in range(1, 5):
            v = scale_lo + (scale_hi - scale_lo) * s / 4
            ax.text(a, s / 4 + 0.03, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#444444", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    chance = [norm(1.0)] * N
    ax.plot(angles + [angles[0]], chance + [chance[0]], color="gray", linewidth=1.4, linestyle="--", zorder=2)
    for mi, m in enumerate(METHODS):
        vals = data[m]
        r = [np.nan if v is None or not np.isfinite(v) else norm(v) for v in vals]
        lw = 3.0 if m == "TwINFER" else 1.5
        ax.plot(angles + [angles[0]], r + [r[0]], color=COLORS[mi], linewidth=lw, zorder=3)
        valid = [(a, rr) for a, rr in zip(angles, r) if np.isfinite(rr)]
        if valid:
            va, vr = zip(*valid)
            ax.scatter(va, vr, color=COLORS[mi], s=18, zorder=4, edgecolors="white", linewidths=0.6)
    handles = [Line2D([0], [0], color=COLORS[i], lw=3.0 if m == "TwINFER" else 1.5,
                      label=PRETTY.get(m, m)) for i, m in enumerate(METHODS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)


def draw_heatmap(cols, grid, blocks_headers, title, out_path):
    """grid: DataFrame methods x columns (AUPRC). blocks_headers: list of (heading, span, [sub headers])."""
    vals = grid.to_numpy(float)
    fig, ax = plt.subplots(figsize=(0.36 * len(cols) + 3.5, 0.42 * len(grid) + 3.0))
    im = ax.imshow(vals, cmap="GnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in range(1, len(cols) + 1)], fontsize=8)
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels([PRETTY.get(m, m) for m in grid.index], fontsize=10)
    ax.get_yticklabels()[0].set_fontweight("bold")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if vals[i, j] > 0.6 else "black")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=34)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Score", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.spines[:].set_visible(False)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)


def base_rate(net_file):
    Mx = np.loadtxt(RW / net_file, delimiter=",")
    n = Mx.shape[0]
    return np.count_nonzero(Mx) / (n * (n - 1))


def real_figures():
    REAL_NETS = [("B_cell", "B_cell_activation", "B_cell.txt"), ("EMT", "EMT", "EMT.txt"), ("GSD", "GSD", "GSD.txt"),
                 ("HSC", "HSC", "HSC.txt"), ("Pluripotent", "Pluripotent", "Pluripotent.txt"),
                 ("VSC", "VSC", "VSC.txt"), ("mCAD", "mCAD", "mCAD.txt")]
    cc = pd.read_csv(SYN / "cross_rho_directional_real.csv").set_index("network")["auprc"]
    summ = pd.read_csv(SUMMARY)
    summ = summ[(summ.variant == "directed_unsigned") & (summ.dataset_id != "ALL")]
    data = {m: [] for m in METHODS}
    names = []
    for label, key, gfile in REAL_NETS:
        names.append(label)
        base = base_rate(gfile)
        for m in METHODS:
            if m == "TwINFER":
                raw = cc[key] if key in cc.index else np.nan
            else:
                r = summ[(summ.dataset_id == key) & (summ.method == SUMMARY_NAME[m])]
                raw = r.auprc.iloc[0] if len(r) else np.nan
            data[m].append(raw / base if np.isfinite(raw) else None)
    all_vals = [v for m in METHODS for v in data[m] if v is not None] + [1.0]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.08
    draw_radar(names, data, max(0.0, lo - pad), hi + pad, "real", REAL_OUT / "cross_corr_scored_auprc_x_radar_real_networks.pdf")
    # table in the same style as the earlier real-network table (bold best per network, grey borders)
    best = []
    for i in range(len(names)):
        cand = [(data[m][i], m) for m in METHODS if data[m][i] is not None]
        best.append(max(cand)[1])
    cell_text = [[("–" if data[m][i] is None else f"{data[m][i]:.2f}") for i in range(len(names))] for m in METHODS]
    labels = [PRETTY.get(m, m) for m in METHODS]
    fig2, ax2 = plt.subplots(figsize=(8, 3.6))
    ax2.axis("off")
    tbl = ax2.table(cellText=cell_text, rowLabels=labels, colLabels=names, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfcfcf")
        if r == 0 or c == -1:
            cell.set_text_props(fontweight="bold")
    for mi, m in enumerate(METHODS):
        for i in range(len(names)):
            if data[m][i] is not None and best[i] == m:
                tbl[(mi + 1, i)].set_text_props(fontweight="bold")
    fig2.savefig(REAL_OUT / "cross_corr_scored_auprc_x_table_real_networks.pdf", format="pdf", bbox_inches="tight",
                 facecolor="none", edgecolor="none", transparent=True)
    plt.close(fig2)
    print("real: wrote radar and table")


def sweep_figures():
    import importlib.util
    spec = importlib.util.spec_from_file_location("P", "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/final_plot_scripts/benchmark/synthetic_network_simulations/plot_auprc_x_sweeps.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    for family in ["network_sweep", "mixed_network_sweep"]:
        p, groups = P.load(family)
        cc = pd.read_csv(SYN / f"cross_rho_directional_{family}.csv")
        cc = cc.set_index("group")["auprc_x"]
        row = [cc.get(g, np.nan) for g in groups_raw(family, p)]
        p2 = p.copy()
        p2.loc["TODO4v2"] = row
        p2 = p2.loc[[m for m in p2.index]]  # keep existing order
        groups2 = list(p2.columns)
        P.LABEL["TODO4v2"] = "TwINFER (cross-correlation)"
        tmp = SYN / "_tmp_radar"
        tmp.mkdir(exist_ok=True)
        P.OUT_DIR = str(tmp)
        P.radar(family, p2, groups2)
        os.replace(tmp / f"auprc_x_radar_{family}.pdf", SYN_OUT / f"cross_corr_scored_auprc_x_radar_{family}.pdf")
        print("sweeps:", family, "radar written")


def groups_raw(family, p):
    """Map the displayed column labels back to raw group names, same order as load()."""
    raw = pd.read_csv(SYN / f"auprc_x_{family}.csv")
    raw = raw[raw.dataset_id.isin(list(P_NET_LABELS))] if family == "network_sweep" else raw
    order = sorted(raw.dataset_id.unique(), key=(lambda s: list(P_NET_LABELS).index(s)) if family == "network_sweep" else mixed_key)
    return order


P_NET_LABELS = {"grn_n6_e5_pos100_density": "5 edges", "grn_n6_e9_pos100_center": "9 edges",
                "n6_e13_pos100": "13 edges", "grn_n6_e9_pos0_sign_ratio": "All negative",
                "grn_n6_e9_pos50_sign_ratio": "Balanced"}


def mixed_key(ds):
    n, e, c, a = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)_pos50", ds).groups()
    return (int(n), int(e), int(c), int(a))


if __name__ == "__main__":
    real_figures()
    sweep_figures()
