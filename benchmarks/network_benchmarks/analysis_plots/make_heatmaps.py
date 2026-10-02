# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Heatmaps (unsigned, directed only) matching heatmap_network_sweep_shared.pdf's
style: methods as rows (sorted by avg rank, best on top), grouped condition
columns with per-topology-rep sub-columns, an Avg + Rank column, for three
metric row-blocks (AUPRC, F1 top-k, F1 threshold).

  1. network_sweep_final, extended with e13 (density sweep at pos100: 5/9/13/17 edges)
  2. mixed_network_sweep (grouped by n_genes x edge_count x autoreg on/off)

"Unsigned directed" = the un-suffixed auprc/f1_topk_all_edges/f1_natural_after_fan_out
(BEELINE: auprc/f1_topk/f1_natural) columns -- score_mixed_network_sweep.py's
_signed and _undirected suffixes are the other two panels from the reference PDF.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
CMAP = "mako_r" if False else None  # placeholder, set below after checking seaborn availability

try:
    import seaborn as sns
    CMAP = sns.color_palette("mako", as_cmap=True)
except ImportError:
    CMAP = plt.get_cmap("viridis")

METHOD_ORDER_FALLBACK = ["TwINFER", "PEARSON", "PPCOR", "PIDC", "GRNBOOST2", "SCSGL", "SCODE", "GENIE3"]

METRIC_BLOCKS = [
    ("AUPRC", "auprc"),
    (r"F$_1$ (top-k)", "f1_topk"),
]

TWINFER_COL = {"auprc": "auprc", "f1_topk": "f1_topk_all_edges", "f1_natural": "f1_natural_after_fan_out"}
BEELINE_COL = {"auprc": "auprc", "f1_topk": "f1_topk", "f1_natural": "f1_natural"}


def build_matrix(twinfer_rows, beeline_rows, columns, metric_key):
    """rows: dict[method] -> {col_label: [values across reps]}. columns: ordered list of col labels."""
    methods = ["TwINFER"] + sorted(beeline_rows.keys())
    data = np.full((len(methods), len(columns)), np.nan)
    for i, m in enumerate(methods):
        src = twinfer_rows if m == "TwINFER" else beeline_rows[m]
        for j, col in enumerate(columns):
            vals = src.get(col, [])
            if vals:
                data[i, j] = np.nanmean(vals)
    avg = np.nanmean(data, axis=1)
    order = np.argsort(-avg)
    methods = [methods[k] for k in order]
    data = data[order]
    avg = avg[order]
    rank = np.arange(1, len(methods) + 1)
    return methods, data, avg, rank


def draw_heatmap(fig_path, title, group_labels, group_sizes, methods_per_block, data_per_block,
                 avg_per_block, rank_per_block, block_titles, sub_labels=None):
    n_blocks = len(data_per_block)
    n_methods = len(methods_per_block[0])
    n_cols = data_per_block[0].shape[1]

    fig_h = 1.6 * n_blocks + 0.5
    fig_w = max(10, 0.62 * n_cols + 3)
    fig, axes = plt.subplots(n_blocks, 1, figsize=(fig_w, fig_h * n_blocks / 1.4))
    if n_blocks == 1:
        axes = [axes]

    for bi, (ax, data, methods, avg, rank, btitle) in enumerate(
        zip(axes, data_per_block, methods_per_block, avg_per_block, rank_per_block, block_titles)
    ):
        full = np.concatenate([data, avg.reshape(-1, 1)], axis=1)
        im = ax.imshow(full, cmap=CMAP, vmin=0, vmax=1, aspect="auto")

        for i in range(full.shape[0]):
            for j in range(full.shape[1]):
                v = full[i, j]
                if np.isnan(v):
                    continue
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=color)
            ax.text(full.shape[1] + 0.65, i, f"#{rank[i]}", ha="left", va="center",
                    fontsize=8, fontweight="bold" if i == 0 else "normal")

        ax.set_yticks(range(n_methods))
        ax.set_yticklabels(methods, fontsize=9)
        for tick, m in zip(ax.get_yticklabels(), methods):
            if m == "TwINFER":
                tick.set_fontweight("bold")

        xticks = list(range(n_cols)) + [n_cols]
        if sub_labels is not None:
            xlabels = list(sub_labels) + ["Avg"]
        else:
            xlabels = [str((j % max(group_sizes)) + 1) if group_sizes else "" for j in range(n_cols)] + ["Avg"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7)

        # group separators + group labels on the top-most block only
        pos = 0
        for gi, (glabel, gsize) in enumerate(zip(group_labels, group_sizes)):
            if pos > 0:
                ax.axvline(pos - 0.5, color="white", linewidth=1.5)
            if bi == 0:
                ax.text(pos + gsize / 2 - 0.5, -1.0, glabel, ha="center", va="bottom", fontsize=8.5)
            pos += gsize
        ax.axvline(n_cols - 0.5, color="white", linewidth=2.5)

        ax.set_ylabel(btitle, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.5, n_cols + 1.3)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.6])
    fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=CMAP), cax=cax, label="Score")
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    for ext in ("png", "pdf"):
        out = fig_path if fig_path.endswith(f".{ext}") else re.sub(r"\.png$", f".{ext}", fig_path)
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print("saved", out)


def make_figure(fig_path, title, group_labels, group_sizes, columns,
                twinfer_rows, beeline_rows, sub_labels=None):
    methods_per_block, data_per_block, avg_per_block, rank_per_block = [], [], [], []
    for _, key in METRIC_BLOCKS:
        methods, data, avg, rank = build_matrix(twinfer_rows[key], beeline_rows[key], columns, key)
        methods_per_block.append(methods)
        data_per_block.append(data)
        avg_per_block.append(avg)
        rank_per_block.append(rank)
    draw_heatmap(fig_path, title, group_labels, group_sizes, methods_per_block, data_per_block,
                avg_per_block, rank_per_block, [b for b, _ in METRIC_BLOCKS], sub_labels=sub_labels)


# =====================================================================
# 1. network_sweep_final (extended with e13), density sweep @ pos100
# =====================================================================
t_old = pd.read_csv(f"{ROOT}/analysis_data/synthetic_network_benchmark_20260824/twinfer_analysis_output.csv")
b_old = pd.read_csv(f"{ROOT}/analysis_data/synthetic_network_benchmark_20260824/beeline_analysis_output.csv")
b_old = b_old[b_old.scheme == "twin_paired"]

t13 = pd.read_csv(f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_analysis_output.csv")
b13 = pd.read_csv(f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output.csv")
b13 = b13[b13.scheme == "twin_paired"]

DENSITY_GROUPS = [
    ("5 edges", ["grn_n6_e5_pos100_density_rep0", "grn_n6_e5_pos100_density_rep1", "grn_n6_e5_pos100_density_rep2"]),
    ("9 edges", ["grn_n6_e9_pos100_center_rep0", "grn_n6_e9_pos100_center_rep1", "grn_n6_e9_pos100_center_rep2"]),
    ("13 edges", ["n6_e13_pos100_rep1", "n6_e13_pos100_rep2", "n6_e13_pos100_rep3"]),
    ("17 edges", ["grn_n6_e17_pos100_density_rep0", "grn_n6_e17_pos100_density_rep1", "grn_n6_e17_pos100_density_rep2"]),
]
columns = [ds for _, dss in DENSITY_GROUPS for ds in dss]
group_labels = [g for g, _ in DENSITY_GROUPS]
group_sizes = [len(dss) for _, dss in DENSITY_GROUPS]

t_all = pd.concat([t_old, t13], ignore_index=True)
b_all = pd.concat([b_old, b13], ignore_index=True)

twinfer_rows = {key: {} for _, key in METRIC_BLOCKS}
beeline_rows = {key: {} for _, key in METRIC_BLOCKS}
for _, key in METRIC_BLOCKS:
    tcol = TWINFER_COL[key]
    for ds in columns:
        vals = t_all.loc[t_all.dataset_id == ds, tcol].dropna().tolist()
        twinfer_rows[key][ds] = vals
    bcol = BEELINE_COL[key]
    for algo in sorted(b_all.algorithm.unique()):
        beeline_rows[key].setdefault(algo, {})
        for ds in columns:
            vals = b_all.loc[(b_all.dataset_id == ds) & (b_all.algorithm == algo), bcol].dropna().tolist()
            beeline_rows[key][algo][ds] = vals

make_figure(
    f"{ROOT}/network_figures/heatmap_network_sweep_final_incl_e13.png",
    "network_sweep_final (incl. e13) -- Unsigned, directed",
    group_labels, group_sizes, columns, twinfer_rows, beeline_rows,
)

# =====================================================================
# 2. mixed_network_sweep, grouped by (n_genes, edge_count) x autoreg on/off
# =====================================================================
tm = pd.read_csv(f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_analysis_output.csv")
bm = pd.read_csv(f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv")
bm = bm[bm.scheme == "twin_paired"]

def parse_mixed(ds):
    m = re.match(r"grn_n(\d+)_e(\d+)_c\d+_a(\d+)_pos\d+_rep\d+", ds)
    n, e, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return n, e, (a > 0)

tm[["n_genes2", "edge_count2", "has_autoreg"]] = tm["dataset_id"].apply(lambda d: pd.Series(parse_mixed(d)))
bm[["n_genes2", "edge_count2", "has_autoreg"]] = bm["dataset_id"].apply(lambda d: pd.Series(parse_mixed(d)))

MIXED_GROUPS = []
for n_genes in [6, 10]:
    edges = sorted(tm.loc[tm.n_genes2 == n_genes, "edge_count2"].unique())
    for e in edges:
        MIXED_GROUPS.append((f"n={n_genes}, e={e}", n_genes, e))

# Top-level grouping is autoregulation status; (n_genes, edge_count) combos are
# the sub-columns within each.
columns_m = []
group_labels_m, group_sizes_m = [], []
for autoreg_label, has_ar in [("Not autoregulated", False), ("Autoregulated", True)]:
    for glabel, n_genes, e in MIXED_GROUPS:
        columns_m.append((n_genes, e, has_ar))
    group_labels_m.append(autoreg_label)
    group_sizes_m.append(len(MIXED_GROUPS))

twinfer_rows_m = {key: {} for _, key in METRIC_BLOCKS}
beeline_rows_m = {key: {} for _, key in METRIC_BLOCKS}
col_labels_m = [f"{n}-{e}-{'a' if a else 'noa'}" for n, e, a in columns_m]
for _, key in METRIC_BLOCKS:
    tcol = TWINFER_COL[key]
    for col_lbl, (n, e, a) in zip(col_labels_m, columns_m):
        sel = tm[(tm.n_genes2 == n) & (tm.edge_count2 == e) & (tm.has_autoreg == a)]
        twinfer_rows_m[key][col_lbl] = sel[tcol].dropna().tolist()
    bcol = BEELINE_COL[key]
    for algo in sorted(bm.algorithm.unique()):
        beeline_rows_m[key].setdefault(algo, {})
        for col_lbl, (n, e, a) in zip(col_labels_m, columns_m):
            sel = bm[(bm.n_genes2 == n) & (bm.edge_count2 == e) & (bm.has_autoreg == a) & (bm.algorithm == algo)]
            beeline_rows_m[key][algo][col_lbl] = sel[bcol].dropna().tolist()

sub_labels_m = [f"n{n},e{e}" for n, e, _ in columns_m]
make_figure(
    f"{ROOT}/network_figures/heatmap_mixed_network_sweep.png",
    "mixed_network_sweep -- Unsigned, directed  (grouped by autoregulation status)",
    group_labels_m, group_sizes_m, col_labels_m, twinfer_rows_m, beeline_rows_m,
    sub_labels=sub_labels_m,
)
