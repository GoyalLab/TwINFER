"""Builds the corrected network_sweep_final and mixed_network_sweep heatmaps: same
layout/competitor data/Avg-Rank convention as before, but TwINFER's numbers are now
this session's from-scratch 'full' formula (see full_formula_v2_network_sweep.py /
full_formula_v2_mixed_network_sweep.py docstrings), not the production package's
shipped twinScore.

Outputs (overwriting the previous, twinScore-based versions):
    network_figures/heatmap_network_sweep_final_incl_e13.{png,pdf}
    network_figures/heatmap_mixed_network_sweep.{png,pdf}
Also writes the assembled wide summary tables next to the TwINFER/competitor CSVs.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: these scripts used to sit in analysis_data/network_sweep_final and read/write their CSVs next to themselves; that data dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from benchmarks.network_benchmarks.score.formula_search.full_formula_v2_common import network_sweep_group_col_to_topology, NETWORK_SWEEP_GROUP_ORDER

FIG_DIR = f'{TWINFER_PROJECT_ROOT}/network_figures'
os.makedirs(FIG_DIR, exist_ok=True)

ALGOS = ["GENIE3", "GRNBOOST2", "PPCOR", "SCODE", "SCSGL", "PEARSON", "PIDC"]

CMAP = LinearSegmentedColormap.from_list(
    "score", ["#f4fbf0", "#a8dba0", "#4fb99f", "#2874a6", "#241468"]
)


def make_panel(ax, mat_df, group_bounds, group_labels, sub_labels, title, vmin=0.0, vmax=1.0):
    """mat_df: rows=algorithm (index), columns=flat sub-columns in display order.
    group_bounds: list of (start_idx, end_idx) exclusive per group, for divider lines.
    """
    piv = mat_df.copy()
    piv["Avg"] = piv.mean(axis=1, skipna=True)
    piv = piv.sort_values("Avg", ascending=False)
    piv["Rank"] = range(1, len(piv) + 1)

    n_rows = len(piv)
    n_cols = mat_df.shape[1]
    scored = piv[list(mat_df.columns) + ["Avg"]].values.astype(float)
    mat = np.hstack([scored, np.full((n_rows, 1), np.nan)])
    n_cols_total = n_cols + 2

    masked = np.ma.masked_invalid(mat)
    cmap = CMAP.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    text_threshold = vmin + 0.62 * (vmax - vmin)
    for i in range(n_rows):
        for j in range(n_cols + 1):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v > text_threshold else "#1a1a1a"
            weight = "bold" if j == n_cols else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color=txt_color, fontweight=weight)
        ax.text(n_cols + 1, i, f"#{piv['Rank'].iloc[i]}", ha="center", va="center",
                 fontsize=8, color="#555")

    ax.set_xticks(range(n_cols_total))
    ax.set_xticklabels(sub_labels + ["Avg", "Rank"], fontsize=7.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(piv.index.tolist(), fontsize=9)
    for i, alg in enumerate(piv.index):
        if alg == "TwINFER":
            ax.get_yticklabels()[i].set_fontweight("bold")

    for start, end in group_bounds:
        if end < n_cols:
            ax.axvline(end - 0.5, color="#888", lw=1.0)
    ax.axvline(n_cols - 0.5, color="#333", lw=1.4)
    ax.axvline(n_cols + 0.5, color="#333", lw=1.4)
    ax.set_xlim(-0.5, n_cols_total - 0.5)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", axis="y", color="white", linewidth=1.2)
    ax.tick_params(which="both", bottom=False, left=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=32)

    # group super-labels above the ticks, in a mixed (data-x, axes-fraction-y)
    # transform so their vertical position doesn't depend on row count and
    # never collides with the title text.
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for (start, end), label in zip(group_bounds, group_labels):
        center = (start + end - 1) / 2.0
        ax.text(center, 1.05, label, ha="center", va="bottom", fontsize=9.5, transform=trans)
    return im


# ============================================================================
# network_sweep_final
# ============================================================================

def build_network_sweep_final():
    twinfer = pd.read_csv(f"{HERE}/full_formula_v2_network_sweep_twinfer.csv")
    comp = pd.read_csv(f"{HERE}/full_formula_v2_network_sweep_competitors.csv")
    col_to_topo = network_sweep_group_col_to_topology()

    sub_cols = [(g, c) for g in NETWORK_SWEEP_GROUP_ORDER for c in (1, 2, 3)]
    sub_labels = [str(c) for _, c in sub_cols]
    group_bounds, group_labels = [], []
    for i, g in enumerate(NETWORK_SWEEP_GROUP_ORDER):
        group_bounds.append((i * 3, i * 3 + 3))
        group_labels.append(g)

    for metric, csv_col in (("auprc", "auprc"), ("f1_topk", "f1_topk")):
        rows = {}
        # TwINFER
        tw_row = {}
        for (g, c) in sub_cols:
            sel = twinfer[(twinfer.group == g) & (twinfer.col == c)]
            tw_row[(g, c)] = float(sel[csv_col].iloc[0]) if len(sel) else np.nan
        rows["TwINFER"] = tw_row
        # competitors
        for algo in ALGOS:
            algo_row = {}
            for (g, c) in sub_cols:
                topo = col_to_topo[(g, c)]
                sel = comp[(comp.dataset_id == topo) & (comp.algorithm == algo)]
                algo_row[(g, c)] = float(sel[csv_col].iloc[0]) if len(sel) else np.nan
            rows[algo] = algo_row
        mat_df = pd.DataFrame(rows).T
        mat_df.columns = pd.MultiIndex.from_tuples(mat_df.columns)
        mat_df = mat_df[sub_cols]
        mat_df.columns = range(len(sub_cols))
        globals()[f"_ns_{metric}"] = mat_df
        mat_df.to_csv(f"{HERE}/full_formula_v2_network_sweep_wide_{metric}.csv")

    fig, axes = plt.subplots(2, 1, figsize=(15, 13))
    fig.subplots_adjust(hspace=0.55, top=0.90)
    im1 = make_panel(axes[0], _ns_auprc, group_bounds, group_labels, sub_labels,
                      "AUPRC", vmin=0.0, vmax=1.0)
    im2 = make_panel(axes[1], _ns_f1_topk, group_bounds, group_labels, sub_labels,
                      "F1 (top-k, tie-aware)", vmin=0.0, vmax=1.0)
    fig.suptitle("network_sweep_final (incl. e13) -- Unsigned, directed\n"
                  "TwINFER = this session's from-scratch 'full' formula (fresh, not the shipped twinScore)",
                  fontsize=12, fontweight="bold", y=0.995)
    cbar = fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Score", fontsize=9)
    out_base = f"{FIG_DIR}/heatmap_network_sweep_final_incl_e13"
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


# ============================================================================
# mixed_network_sweep
# ============================================================================

MIXED_COLUMNS = [
    ("Not autoregulated", "6 genes,\n6 edges", 6, 6, "a0"),
    ("Not autoregulated", "6 genes,\n12 edges", 6, 12, "a0"),
    ("Not autoregulated", "10 genes,\n10 edges", 10, 10, "a0"),
    ("Not autoregulated", "10 genes,\n20 edges", 10, 20, "a0"),
    ("Autoregulated", "6 genes,\n6 edges", 6, 6, "a3"),
    ("Autoregulated", "6 genes,\n12 edges", 6, 12, "a3"),
    ("Autoregulated", "10 genes,\n10 edges", 10, 10, "a5"),
    ("Autoregulated", "10 genes,\n20 edges", 10, 20, "a5"),
]


def topologies_for_mixed_column(n, e, autoreg_tag):
    cycles = (3, 4) if n == 6 else (5, 8)
    names = []
    for c in cycles:
        for rep in (0, 1, 2):
            names.append(f"grn_n{n}_e{e}_c{c}_{autoreg_tag}_pos50_rep{rep}")
    return names


def build_mixed_network_sweep():
    twinfer = pd.read_csv(f"{HERE}/full_formula_v2_mixed_network_sweep_twinfer.csv").set_index("topology")
    comp = pd.read_csv(f"{HERE}/full_formula_v2_mixed_network_sweep_competitors.csv")

    sub_labels_top = [c for _, c, *_ in MIXED_COLUMNS[:4]]
    group_bounds = [(0, 4), (4, 8)]
    group_labels = ["Not autoregulated", "Autoregulated"]

    for metric, csv_col in (("auprc", "auprc"), ("f1_topk", "f1_topk")):
        rows = {}
        tw_row = {}
        for (_, label, n, e, tag) in MIXED_COLUMNS:
            names = topologies_for_mixed_column(n, e, tag)
            vals = twinfer.loc[twinfer.index.intersection(names), csv_col]
            tw_row[label + ("|A" if tag in ("a3", "a5") else "|N")] = float(vals.mean()) if len(vals) else np.nan
        rows["TwINFER"] = tw_row
        for algo in ALGOS:
            algo_row = {}
            for (_, label, n, e, tag) in MIXED_COLUMNS:
                names = topologies_for_mixed_column(n, e, tag)
                sel = comp[(comp.dataset_id.isin(names)) & (comp.algorithm == algo)]
                algo_row[label + ("|A" if tag in ("a3", "a5") else "|N")] = float(sel[csv_col].mean()) if len(sel) else np.nan
            rows[algo] = algo_row
        mat_df = pd.DataFrame(rows).T
        # order columns: Not-autoreg x4 then Autoreg x4
        ordered_cols = [f"{label}|N" for _, label, *_ in MIXED_COLUMNS[:4]] + \
                       [f"{label}|A" for _, label, *_ in MIXED_COLUMNS[4:]]
        mat_df = mat_df[ordered_cols]
        mat_df.columns = range(8)
        globals()[f"_mx_{metric}"] = mat_df
        mat_df.to_csv(f"{HERE}/full_formula_v2_mixed_network_sweep_wide_{metric}.csv")

    fig, axes = plt.subplots(2, 1, figsize=(12, 13))
    fig.subplots_adjust(hspace=0.55, top=0.90)
    im1 = make_panel(axes[0], _mx_auprc, group_bounds, group_labels, sub_labels_top * 2,
                      "AUPRC", vmin=0.0, vmax=1.0)
    im2 = make_panel(axes[1], _mx_f1_topk, group_bounds, group_labels, sub_labels_top * 2,
                      "F1 (top-k, tie-aware)", vmin=0.0, vmax=1.0)
    fig.suptitle("mixed_network_sweep -- Unsigned, directed  (grouped by autoregulation status)\n"
                  "TwINFER = this session's from-scratch 'full' formula (fresh, not the shipped twinScore)",
                  fontsize=12, fontweight="bold", y=0.995)
    cbar = fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Score", fontsize=9)
    out_base = f"{FIG_DIR}/heatmap_mixed_network_sweep"
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


REAL_DATA_NETS = ["GSD", "HSC", "VSC", "mCAD", "B_cell_activation", "EMT", "Pluripotent"]
REAL_DATA_ID_MAP = {  # our twinfer csv's dataset names -> beeline_scores.csv's dataset_id
    "GSD": "GSD", "HSC_balanced": "HSC", "VSC": "VSC", "mCAD": "mCAD",
    "B_cell_activation": "B_cell_activation", "EMT": "EMT", "Pluripotent": "Pluripotent",
}


def build_real_data():
    twinfer = pd.read_csv(f"{HERE.replace('/network_sweep_final', '/paper_analysis/real_networks')}/full_formula_v2_real_data_twinfer.csv")
    twinfer["net"] = twinfer.dataset.str.replace("real_data:", "", regex=False).map(REAL_DATA_ID_MAP)
    comp = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv')
    comp = comp[comp.scheme == "spread"]

    for metric, tw_col, comp_col in (("auprc", "auprc_full", "auprc"), ("f1_topk", "f1_full", "f1_topk")):
        rows = {}
        tw_row = {net: float(twinfer.loc[twinfer.net == net, tw_col].mean()) for net in REAL_DATA_NETS}
        rows["TwINFER"] = tw_row
        for algo in ALGOS:
            algo_row = {}
            for net in REAL_DATA_NETS:
                sel = comp[(comp.dataset_id == net) & (comp.algorithm == algo)]
                algo_row[net] = float(sel[comp_col].mean()) if len(sel) else np.nan
            rows[algo] = algo_row
        mat_df = pd.DataFrame(rows).T
        mat_df = mat_df[REAL_DATA_NETS]
        globals()[f"_rd_{metric}"] = mat_df
        mat_df.to_csv(f"{HERE}/full_formula_v2_real_data_wide_{metric}.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11, 12))
    fig.subplots_adjust(hspace=0.55, top=0.90)
    im1 = make_panel(axes[0], _rd_auprc, [(0, 7)], ["Real-world networks"], REAL_DATA_NETS,
                      "AUPRC", vmin=0.0, vmax=1.0)
    im2 = make_panel(axes[1], _rd_f1_topk, [(0, 7)], ["Real-world networks"], REAL_DATA_NETS,
                      "F1 (top-k, tie-aware)", vmin=0.0, vmax=1.0)
    fig.suptitle("real_data -- Unsigned, directed\n"
                  "TwINFER = this session's from-scratch 'full' formula (fresh, not the shipped twinScore)",
                  fontsize=12, fontweight="bold", y=0.995)
    cbar = fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Score", fontsize=9)
    out_base = f"{FIG_DIR}/heatmap_real_data"
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


LARRY_ALGOS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
LARRY_DISPLAY = {"rho": "rho", "ppcor": "PPCOR", "pidc": "PIDC", "genie3": "GENIE3", "grnboost2": "GRNBoost2"}
LARRY_GROUPS = [
    ("Variability", "high", "variability_high"), ("Variability", "mid", "variability_mid"), ("Variability", "low", "variability_low"),
    ("Detection", "high", "detection_high"), ("Detection", "mid", "detection_mid"), ("Detection", "low", "detection_low"),
    ("Correlation", "high", "correlation_high"), ("Correlation", "mid", "correlation_mid"), ("Correlation", "low", "correlation_low"),
]


def build_larry():
    LARRY_HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
    # [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] twinfer = pd.read_csv(f"{LARRY_HERE}/full_formula_v2_larry_twinfer.csv").set_index("gene_set")
    twinfer = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/full_formula_v2_larry_twinfer.csv").set_index("gene_set")
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] bench = pd.read_csv(f"{LARRY_HERE}/resources/benchmark/benchmark_results.csv")
    bench = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/benchmark/benchmark_results.csv")

    rows = {}
    rows["TwINFER"] = {gs: float(twinfer.loc[gs, "auprc_full"]) for _, _, gs in LARRY_GROUPS}
    for algo in LARRY_ALGOS:
        algo_row = {}
        for _, _, gs in LARRY_GROUPS:
            sel = bench[(bench.gene_set == gs) & (bench.method == algo)]
            algo_row[gs] = float(sel.auprc.iloc[0]) if len(sel) else np.nan
        rows[LARRY_DISPLAY[algo]] = algo_row
    mat_df = pd.DataFrame(rows).T
    mat_df = mat_df[[gs for _, _, gs in LARRY_GROUPS]]
    mat_df.columns = range(9)
    mat_df.to_csv(f"{HERE}/full_formula_v2_larry_wide_auprc.csv")

    sub_labels = [lvl for _, lvl, _ in LARRY_GROUPS]
    group_bounds = [(0, 3), (3, 6), (6, 9)]
    group_labels = ["Variability", "Detection", "Correlation"]

    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    fig.subplots_adjust(top=0.82)
    im = make_panel(ax, mat_df, group_bounds, group_labels, sub_labels,
                     "AUPRC", vmin=0.0, vmax=1.0)
    fig.suptitle("LARRY (9 curated gene-set panels) -- Unsigned, directed\n"
                  "TwINFER = this session's from-scratch 'full' formula (fresh, not the shipped twinScore)",
                  fontsize=12, fontweight="bold", y=0.98)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Score", fontsize=9)
    out_base = f"{FIG_DIR}/heatmap_larry"
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


def build_larry_ratio():
    LARRY_HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
    # [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] twinfer = pd.read_csv(f"{LARRY_HERE}/full_formula_v2_larry_twinfer.csv").set_index("gene_set")
    twinfer = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/full_formula_v2_larry_twinfer.csv").set_index("gene_set")
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] bench = pd.read_csv(f"{LARRY_HERE}/resources/benchmark/benchmark_results.csv")
    bench = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/benchmark/benchmark_results.csv")

    random_baseline = {gs: bench[bench.gene_set == gs].auprc_random.mean() for _, _, gs in LARRY_GROUPS}

    rows = {}
    rows["TwINFER"] = {gs: float(twinfer.loc[gs, "auprc_full"]) / random_baseline[gs] for _, _, gs in LARRY_GROUPS}
    for algo in LARRY_ALGOS:
        algo_row = {}
        for _, _, gs in LARRY_GROUPS:
            sel = bench[(bench.gene_set == gs) & (bench.method == algo)]
            algo_row[gs] = float(sel.auprc.iloc[0]) / random_baseline[gs] if len(sel) else np.nan
        rows[LARRY_DISPLAY[algo]] = algo_row
    mat_df = pd.DataFrame(rows).T
    mat_df = mat_df[[gs for _, _, gs in LARRY_GROUPS]]
    mat_df.columns = range(9)
    mat_df.to_csv(f"{HERE}/full_formula_v2_larry_wide_ratio.csv")

    sub_labels = [lvl for _, lvl, _ in LARRY_GROUPS]
    group_bounds = [(0, 3), (3, 6), (6, 9)]
    group_labels = ["Variability", "Detection", "Correlation"]

    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    fig.subplots_adjust(top=0.82)
    im = make_panel(ax, mat_df, group_bounds, group_labels, sub_labels,
                     "AUPRC / random-baseline AUPRC", vmin=0.7, vmax=2.0)
    fig.suptitle("LARRY (9 curated gene-set panels) -- ratio to random baseline, unsigned/directed\n"
                  "TwINFER = this session's from-scratch 'full' formula (fresh, not the shipped twinScore)",
                  fontsize=12, fontweight="bold", y=0.98)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Ratio to random", fontsize=9)
    out_base = f"{FIG_DIR}/heatmap_larry_ratio"
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


def build_larry_precision():
    """Precision at top-k (k=n_true_edges): identical to F1-top-k in this convention since
    selecting exactly k items with no ties makes precision=recall=F1 -- verified directly
    against benchmark_results.csv's early_precision column (== f1 exactly, every row)."""
    LARRY_HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
    # [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] twinfer = pd.read_csv(f"{LARRY_HERE}/full_formula_v2_larry_twinfer.csv").set_index("gene_set")
    twinfer = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/full_formula_v2_larry_twinfer.csv").set_index("gene_set")
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] bench = pd.read_csv(f"{LARRY_HERE}/resources/benchmark/benchmark_results.csv")
    bench = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/benchmark/benchmark_results.csv")

    random_baseline = {gs: bench[bench.gene_set == gs].f1_random.mean() for _, _, gs in LARRY_GROUPS}

    rows_raw, rows_ratio = {}, {}
    rows_raw["TwINFER"] = {gs: float(twinfer.loc[gs, "f1_full"]) for _, _, gs in LARRY_GROUPS}
    rows_ratio["TwINFER"] = {gs: float(twinfer.loc[gs, "f1_full"]) / random_baseline[gs] for _, _, gs in LARRY_GROUPS}
    for algo in LARRY_ALGOS:
        raw_row, ratio_row = {}, {}
        for _, _, gs in LARRY_GROUPS:
            sel = bench[(bench.gene_set == gs) & (bench.method == algo)]
            v = float(sel.early_precision.iloc[0]) if len(sel) else np.nan
            raw_row[gs] = v
            ratio_row[gs] = v / random_baseline[gs] if len(sel) else np.nan
        rows_raw[LARRY_DISPLAY[algo]] = raw_row
        rows_ratio[LARRY_DISPLAY[algo]] = ratio_row

    mat_raw = pd.DataFrame(rows_raw).T[[gs for _, _, gs in LARRY_GROUPS]]
    mat_ratio = pd.DataFrame(rows_ratio).T[[gs for _, _, gs in LARRY_GROUPS]]
    mat_raw.columns = range(9)
    mat_ratio.columns = range(9)
    mat_raw.to_csv(f"{HERE}/full_formula_v2_larry_wide_precision.csv")
    mat_ratio.to_csv(f"{HERE}/full_formula_v2_larry_wide_precision_ratio.csv")

    sub_labels = [lvl for _, lvl, _ in LARRY_GROUPS]
    group_bounds = [(0, 3), (3, 6), (6, 9)]
    group_labels = ["Variability", "Detection", "Correlation"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 13))
    fig.subplots_adjust(hspace=0.55, top=0.90)
    im1 = make_panel(axes[0], mat_raw, group_bounds, group_labels, sub_labels,
                      "Precision (top-k, k=n_true_edges)", vmin=0.0, vmax=1.0)
    im2 = make_panel(axes[1], mat_ratio, group_bounds, group_labels, sub_labels,
                      "Precision / random-baseline precision", vmin=0.5, vmax=2.5)
    fig.suptitle("LARRY (9 curated gene-set panels) -- precision at top-k, unsigned/directed\n"
                  "TwINFER = this session's from-scratch 'full' formula (fresh, not the shipped twinScore)",
                  fontsize=12, fontweight="bold", y=0.995)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.025, pad=0.02)
    cbar1.set_label("Precision", fontsize=9)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.025, pad=0.02)
    cbar2.set_label("Ratio to random", fontsize=9)
    out_base = f"{FIG_DIR}/heatmap_larry_precision"
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


if __name__ == "__main__":
    build_network_sweep_final()
    build_mixed_network_sweep()
    build_real_data()
    build_larry()
    build_larry_ratio()
    build_larry_precision()
