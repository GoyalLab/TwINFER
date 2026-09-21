"""Static PDF+PNG heatmap built DIRECTLY from summary_metrics_table_network_sweep_t1_1.csv (the
canonical, already-reviewed network_sweep summary table -- variant, dataset_id, method, f1_topk,
precision_topk, auprc). That table is aggregated to ONE value per topology family (no
per-replicate breakdown), so this heatmap has one column per topology, not 3 replicate
sub-columns like the earlier per-replicate build.

3 variants (Signed directed / Unsigned directed / Unsigned undirected) x 2 metrics (AUPRC /
F1 top-k) = 6 panels. Columns grouped: Network density (5/9/13 edges) and Fraction of negative
edges (all-negative/balanced/all-positive, 9-edge topology reused as all-positive -- same value,
consistent with how these topologies double as both a density point and a sign-ratio endpoint).
e17_pos100_density excluded per instruction. Colors/font match the user-provided reference: pale
mint-green (low) -> teal -> dark navy blue (high), bold sans-serif cell text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

HERE = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark"
DF = pd.read_csv(f"{HERE}/summary_metrics_table_network_sweep_t1_1.csv")

GROUPS = [
    ("Network density", [("5 edges", "grn_n6_e5_pos100_density"),
                          ("9 edges", "grn_n6_e9_pos100_center"),
                          ("13 edges", "n6_e13_pos100")]),
    ("Fraction of negative edges", [("All negative", "grn_n6_e9_pos0_sign_ratio"),
                                     ("Balanced", "grn_n6_e9_pos50_sign_ratio"),
                                     ("All positive", "grn_n6_e9_pos100_center")]),
]
PANELS = [
    ("signed_directed", "Signed directed"),
    ("directed_unsigned", "Unsigned directed"),
    ("undirected", "Unsigned undirected"),
]
METRICS = [("auprc", "AUPRC"), ("f1_topk", "F1 (top-k)")]
METHOD_ORDER = ["TODO4v2", "PEARSON", "PPCOR", "PIDC", "GRNBOOST2", "SCSGL", "SCODE", "GENIE3"]
DISPLAY_NAME = {"TODO4v2": "TODO4v2", "PEARSON": "Pearson", "PPCOR": "PPCOR", "PIDC": "PIDC",
                 "GRNBOOST2": "GRNBoost2", "SCSGL": "scSGL", "SCODE": "SCODE", "GENIE3": "GENIE3"}

# pale mint-green -> teal -> dark navy blue, matching the user-provided reference image
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "mint_teal_navy", ["#d9f2dc", "#a9e0b4", "#5fbf9a", "#2f9a9a", "#2c6fa0", "#1f4e8c"])

FONT = "DejaVu Sans"
N_COLS = sum(len(cols) for _, cols in GROUPS)  # 6

Y_TITLE = -5.7
Y_GROUP = -4.9
Y_SUBGROUP_ANCHOR = -0.25  # rotated labels grow upward (more negative) from here
HEADER_TOP = -6.3


def panel_values(variant, metric):
    sub = DF[(DF.variant == variant) & (DF.dataset_id != "ALL")]
    rows = []
    for method in METHOD_ORDER:
        m = sub[sub.method == method].set_index("dataset_id")[metric]
        vals = []
        for _, cols in GROUPS:
            for _, dsid in cols:
                vals.append(float(m.get(dsid, float("nan"))))
        avg = sum(vals) / len(vals)
        rows.append((method, vals, avg))
    rows.sort(key=lambda r: -r[2])
    return rows


def draw_panel(ax, variant, title, metric):
    rows = panel_values(variant, metric)
    n_rows = len(rows)
    mat = [r[1] for r in rows]

    im = ax.imshow(mat, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
                    extent=(0, N_COLS, n_rows, 0), zorder=1)
    for i in range(n_rows):
        for j in range(N_COLS):
            v = mat[i][j]
            color = "white" if v >= 0.5 else "#17140f"
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                     fontsize=8.2, color=color, family=FONT, fontweight="bold", zorder=2)

    col = 0
    for _, cols in GROUPS:
        ax.axvline(col, color="white", lw=1.6, zorder=3)
        col += len(cols)
    for i in range(n_rows + 1):
        ax.axhline(i, color="white", lw=1.0, zorder=3)

    ax.set_xticks([])
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels([DISPLAY_NAME[m] for m, _, _ in rows], fontsize=8.4, family=FONT)
    for lbl, (method, _, _) in zip(ax.get_yticklabels(), rows):
        if method == "TODO4v2":
            lbl.set_fontweight("bold")

    AVG_X, AVG_W, RANK_X = N_COLS + 0.4, 1.4, N_COLS + 2.1
    for i, (method, _, avg) in enumerate(rows):
        ax.add_patch(plt.Rectangle((AVG_X, i), AVG_W, 1, color=CMAP(avg), zorder=1))
        txt_color = "white" if avg >= 0.5 else "#17140f"
        ax.text(AVG_X + AVG_W / 2, i + 0.5, f"{avg:.2f}", ha="center", va="center",
                 fontsize=8.2, fontweight="bold", color=txt_color, family=FONT, zorder=2)
        ax.text(RANK_X, i + 0.5, f"#{i+1}", ha="left", va="center",
                 fontsize=7.6, color="#8a8570", family=FONT)
    ax.text(AVG_X + AVG_W / 2, Y_SUBGROUP, "Avg.", ha="center", va="center",
             fontsize=7.6, fontweight="bold", color="#17140f", family=FONT)
    ax.text(RANK_X, Y_SUBGROUP, "Rank", ha="left", va="center",
             fontsize=7.6, fontweight="bold", color="#17140f", family=FONT)

    for i, (method, _, _) in enumerate(rows):
        if method == "TODO4v2":
            ax.add_patch(plt.Rectangle((-0.3, i), 0.3, 1, color="#c98500",
                                        transform=ax.transData, clip_on=False, zorder=4))

    col = 0
    for _, cols in GROUPS:
        for clabel, _ in cols:
            ax.text(col + 0.5, Y_SUBGROUP, clabel, ha="center", va="center",
                     fontsize=7.4, fontweight="bold", color="#55503f", family=FONT)
            col += 1

    x0 = 0
    for glabel, cols in GROUPS:
        span = len(cols)
        ax.text(x0 + span / 2, Y_GROUP, glabel, ha="center", va="center",
                 fontsize=8.0, fontweight="bold", color="#17140f", family=FONT)
        x0 += span

    ax.text(0, Y_TITLE, f"{title} — {metric_label(metric)}", ha="left", va="center",
             fontsize=10.5, fontweight="bold", color="#17140f", family=FONT)

    ax.set_xlim(0, N_COLS + 3.1)
    ax.set_ylim(n_rows, HEADER_TOP)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def metric_label(metric):
    return dict(METRICS)[metric]


def render():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.87, bottom=0.03, hspace=0.2, wspace=0.4)

    last_im = None
    for row_idx, (metric, _) in enumerate(METRICS):
        for col_idx, (variant, title) in enumerate(PANELS):
            last_im = draw_panel(axes[row_idx, col_idx], variant, title, metric)

    fig.text(0.05, 0.975, "network_sweep: TODO4v2 vs. 7 BEELINE competitors",
              fontsize=15, fontweight="bold", family=FONT, color="#17140f")
    fig.text(0.05, 0.952,
              "t1=1. One column per topology family (source: summary_metrics_table_network_"
              "sweep_t1_1.csv). e17_pos100_density excluded. 9-edge topology reused as the "
              "‘all positive’ negative-fraction column.",
              fontsize=9, family=FONT, color="#55503f")

    cax = fig.add_axes((0.83, 0.955, 0.15, 0.015))
    cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.set_label("score", fontsize=8, family=FONT, labelpad=2)
    return fig


def main():
    fig = render()
    fig.savefig(f"{HERE}/heatmap_network_sweep.pdf", format="pdf", bbox_inches="tight")
    fig2 = render()
    fig2.savefig(f"{HERE}/heatmap_network_sweep.png", format="png", dpi=200, bbox_inches="tight")
    print("wrote heatmap_network_sweep.pdf / .png")


if __name__ == "__main__":
    main()
