"""AUPRC-x table (heatmap-style) and radar for the synthetic sweeps, same style as the real-network figures
(plot_radar_and_table_auprc_x.py). Input: analysis_data/synthetic_network_simulations/auprc_x_<family>.csv
written by score_auprc_x_sweeps.py (TwINFER = TODO4v2 nogate, directed unsigned, AUPRC-x = AUPRC / base rate).

Date:   2026-10-04
Output: final_plots/benchmark/synthetic_network_simulations/auprc_x_{table,radar}_<family>.pdf (vector, transparent)
"""
import os
import re
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

IN_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/synthetic_network_simulations"
OUT_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/final_plots/benchmark/synthetic_network_simulations"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {"TODO4v2": "#2a78d6", "GENIE3": "#eb6834", "GRNBOOST2": "#1baf7a", "PEARSON": "#eda100",
          "PIDC": "#e87ba4", "PPCOR": "#008300", "SCODE": "#4a3aa7", "SCSGL": "#e34948"}
LABEL = {"TODO4v2": "TwINFER (TODO4v2)", "GRNBOOST2": "GRNBoost2"}
FAMILIES = {"network_sweep": "network sweep", "mixed_network_sweep": "mixed network sweep"}


NETWORK_LABEL = {"grn_n6_e5_pos100_density": "5 edges", "grn_n6_e9_pos100_center": "9 edges",
                 "n6_e13_pos100": "13 edges", "grn_n6_e9_pos0_sign_ratio": "All negative",
                 "grn_n6_e9_pos50_sign_ratio": "Balanced"}


def mixed_key(ds):
    """Heatmap block order: 6g6e, 6g12e, 10g10e, 10g20e; then cycle, then a-level."""
    n, e, c, a = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)_pos50", ds).groups()
    return (int(n), int(e), int(c), int(a))


def axis_label(family, ds):
    """Axis names as in the heatmaps: density / fraction blocks for network_sweep, genes-edges + cycle + a-level for mixed."""
    if family == "network_sweep":
        return NETWORK_LABEL[ds]
    m = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)_pos50", ds)
    n, e, c, a = m.groups()
    autoreg = "No autoregulation" if a == "0" else "Autoregulation"
    return f"{n} genes, {e} edges\nCycle {c}\n{autoreg}"


def load(family):
    d = pd.read_csv(f"{IN_DIR}/auprc_x_{family}.csv")
    if family == "network_sweep":
        d = d[d.dataset_id.isin(NETWORK_LABEL)]
    p = d.pivot(index="method", columns="dataset_id", values="auprc_x")
    if family == "network_sweep":
        p = p[list(NETWORK_LABEL)]
    else:
        p = p[sorted(p.columns, key=mixed_key)]
    order = p.mean(axis=1).sort_values(ascending=False).index.tolist()
    p = p.loc[order]
    p.columns = [axis_label(family, c) for c in p.columns]
    return p, p.columns.tolist()


def table(family, p, groups):
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(groups) + 2.5), 3.8))
    vals = p.to_numpy(dtype=float)
    vmin, vmax = 0.0, np.nanmax(vals)
    im = ax.imshow(vals, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace("grn_", "").replace("_pos100", "").replace("_pos50", "") for g in groups],
                       rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels([LABEL.get(m, m) for m in p.index], fontsize=9)
    best = np.nanargmax(vals, axis=0)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color="#444444")
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if best[j] == i else "normal",
                    color="white" if v > 0.6 * vmax else "black")
    ax.set_title(f"AUPRC× – {FAMILIES[family]}", fontsize=11, fontweight="bold", pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("AUPRC×  (1.0 = chance)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(f"{OUT_DIR}/auprc_x_table_{family}.pdf", format="pdf", bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)


def radar(family, p, groups):
    N = len(groups)
    angles = [2 * np.pi * i / N for i in range(N)]
    all_vals = [v for v in p.to_numpy(dtype=float).ravel() if np.isfinite(v)] + [1.0]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.08
    lo, hi = max(0.0, lo - pad), hi + pad

    def norm(v):
        return (v - lo) / (hi - lo)

    fig = plt.figure(figsize=(7.0, 7.4))
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
        ax.text(a, 1.26, groups[i], ha="center", va="center", fontsize=8, color="black", linespacing=1.1)
        for st in range(1, 5):
            v = lo + (hi - lo) * st / 4
            ax.text(a, st / 4 + 0.03, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#444444", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    chance = [norm(1.0)] * N
    ax.plot(angles + [angles[0]], chance + [chance[0]], color="gray", linewidth=1.4, linestyle="--", zorder=2)
    for m in p.index:
        vals = p.loc[m].to_numpy(dtype=float)
        r = [np.nan if not np.isfinite(v) else norm(v) for v in vals]
        lw = 3.0 if m == "TODO4v2" else 1.5
        ax.plot(angles + [angles[0]], r + [r[0]], color=COLORS.get(m, "#777777"), linewidth=lw, zorder=3)
        valid = [(a, rr) for a, rr in zip(angles, r) if np.isfinite(rr)]
        if valid:
            va, vr = zip(*valid)
            ax.scatter(va, vr, color=COLORS.get(m, "#777777"), s=18, zorder=4, edgecolors="white", linewidths=0.6)
    handles = [Line2D([0], [0], color=COLORS.get(m, "#777777"), lw=3.0 if m == "TODO4v2" else 1.5,
                      label=LABEL.get(m, m)) for m in p.index]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.savefig(f"{OUT_DIR}/auprc_x_radar_{family}.pdf", format="pdf", bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    for family in FAMILIES:
        p, groups = load(family)
        table(family, p, groups)
        radar(family, p, groups)
        print("wrote", family)
