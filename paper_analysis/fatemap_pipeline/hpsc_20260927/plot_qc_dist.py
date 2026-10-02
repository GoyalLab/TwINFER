#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""QC distribution plots (total UMI counts, detected genes, mito%) for
endo_T0 vs endo_T1, from qc_metrics.csv (computed by compute_qc_metrics.py).
Step-histogram outlines, log-x for count-based metrics, validated categorical
pair (blue #2a78d6 / orange #eb6834) per the dataviz skill's default theme.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/qc_metrics.csv'
OUT = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/qc_distributions.png'

COLORS = {"endo_T0": "#2a78d6", "endo_T1": "#eb6834"}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

MIN_GENES = 1000
MAX_MITO = 10.0

df = pd.read_csv(CSV)
df["pass_qc"] = (df["n_genes"] >= MIN_GENES) & (df["pct_counts_mt"] <= MAX_MITO)

fig, axes = plt.subplots(2, 3, figsize=(15, 8.0))

panels = [
    ("total_counts", "Total UMI counts / cell", True),
    ("n_genes", "Detected genes / cell", True),
    ("pct_counts_mt", "Mitochondrial % / cell", False),
]


def draw_row(ax_row, data, row_title):
    for ax, (col, title, logx) in zip(ax_row, panels):
        for sample, color in COLORS.items():
            vals = data.loc[data["sample"] == sample, col].to_numpy()
            vals = vals[vals > 0] if logx else vals
            if logx:
                lo, hi = max(vals.min(), 1), vals.max()
                n_bins = 40
                while (lo * (hi / lo) ** (1 / n_bins) - lo) < 1 and n_bins > 5:
                    n_bins -= 1
                bins = np.logspace(np.log10(lo), np.log10(hi), n_bins)
            else:
                bins = np.linspace(0, min(vals.max(), 30), 31)
            ax.hist(vals, bins=bins, histtype="step", linewidth=1.5, color=color, label=sample, density=False)
        if logx:
            ax.set_xscale("log")
        ax.set_title(f"{row_title}: {title}", fontsize=10.5, color=INK)
        ax.set_ylabel("number of cells", fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        if col == "pct_counts_mt":
            ax.axvline(MAX_MITO, color=MUTED, linewidth=1, linestyle="--")
        if col == "n_genes":
            ax.axvline(MIN_GENES, color=MUTED, linewidth=1, linestyle="--")


draw_row(axes[0], df, "Raw (Cell Ranger, pre-threshold)")
draw_row(axes[1], df[df["pass_qc"]], "Thresholded (n_genes≥%d, mito≤%d%%)" % (MIN_GENES, MAX_MITO))

n_raw = df.groupby("sample").size()
n_pass = df[df["pass_qc"]].groupby("sample").size()
legend_labels = {
    s: f"{s}  (n={n_raw[s]:,} → {n_pass[s]:,})" for s in COLORS
}
for ax_row in axes:
    handles = ax_row[0].get_legend_handles_labels()[0]
    ax_row[0].legend(handles, [legend_labels[s] for s in COLORS], frameon=False, fontsize=8.5, labelcolor=INK)

fig.suptitle(
    "endo_T0 vs endo_T1 — per-cell QC, raw vs thresholded (Cell Ranger filtered matrix, pre-singletCode)",
    fontsize=12, color=INK,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=150, facecolor="#fcfcfb")
print(f"wrote {OUT}")
print(f"\nn_raw: {dict(n_raw)}")
print(f"n_pass_qc: {dict(n_pass)}")
