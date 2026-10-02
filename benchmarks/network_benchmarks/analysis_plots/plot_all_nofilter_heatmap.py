"""
Heatmap comparison of the combined old+new, filtered+no-filter TwINFER-vs-BEELINE
results (real_networks_all_nofilter_combined_scores.csv), styled like the
reference autoregulation heatmap: rows = methods sorted by average score
(best at top, TwINFER rows bold), columns = datasets, trailing Avg + Rank
columns, cells annotated with the numeric score, sequential light-green ->
dark-purple colormap.

Layout: the original 8 real networks and the 8 multistate/seeded datasets are
on separate pages, and the unsigned and signed method sets are on separate
pages too (BEELINE has no signed scoring, so its rows are identical in both --
they are the shared reference baseline; only the TwINFER rows differ). Four
metrics: raw AUPRC, raw F1 (top-k), and each divided by the dataset's random
baseline (= its true-edge prevalence). So 2 dataset-groups x 2 sign-modes x
4 metrics = 16 pages.

Output: real_networks_all_nofilter_heatmap.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from twinfer.utils.paths import get_data_root

DATA_ROOT = get_data_root()
REAL_NET_ROOT = DATA_ROOT / "paper_analysis" / "real_networks"

ORIGINAL_DATASETS = ["GSD", "HSC", "VSC", "mCAD", "EMT", "Pluripotent", "B_cell_activation"]
MULTISTATE_DATASETS = ["GSD_multistate", "GSD_seeded", "HSC_multistate", "HSC_seeded",
                       "EMT_multistate", "EMT_seeded", "VSC_seeded", "mCAD_seeded"]
COLUMN_LABELS = {"B_cell_activation": "B_cell", "Circadian_cycle": "Circadian"}

BEELINE = ["PIDC", "GENIE3", "GRNBOOST2", "PPCOR", "SCODE", "SCSGL", "PEARSON"]
BEELINE_SIGNED = [f"{b} (signed)" for b in BEELINE]
# Only the no-filter + z_het-fixed twinScore variant is shown, labelled simply
# "TwINFER" (see RENAME). The other variants stay in the CSV for reference.
TWINFER_UNSIGNED = ["TwINFER (no-filter, z_het-fixed)"]
TWINFER_SIGNED = ["TwINFER (no-filter, z_het-fixed, signed)"]
RENAME = {
    "TwINFER (no-filter, z_het-fixed)": "TwINFER",
    "TwINFER (no-filter, z_het-fixed, signed)": "TwINFER",
    **{f"{b} (signed)": b for b in BEELINE},
}

CMAP = LinearSegmentedColormap.from_list(
    "score", ["#f4fbf0", "#a8dba0", "#4fb99f", "#2874a6", "#241468"]
)


def make_panel(ax, df, metric, columns, title, vmin=0.0, vmax=1.0, fmt="{:.2f}"):
    piv = df.pivot(index="algorithm", columns="dataset", values=metric)
    piv = piv.rename(index=RENAME)
    piv = piv.reindex(columns=columns)
    piv["Avg"] = piv.mean(axis=1, skipna=True)
    piv = piv.sort_values("Avg", ascending=False)
    piv["Rank"] = range(1, len(piv) + 1)

    n_rows = len(piv)
    n_cols = len(columns)
    scored = piv[columns + ["Avg"]].values.astype(float)
    mat = np.hstack([scored, np.full((n_rows, 1), np.nan)])   # trailing Rank col left blank
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
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                     color=txt_color, fontweight=weight)
        ax.text(n_cols + 1, i, f"#{piv['Rank'].iloc[i]}", ha="center", va="center",
                 fontsize=8, color="#555")

    ax.set_xticks(range(n_cols_total))
    ax.set_xticklabels([COLUMN_LABELS.get(c, c) for c in columns] + ["Avg", "Rank"],
                        rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(piv.index.tolist(), fontsize=9)
    for i, alg in enumerate(piv.index):
        if alg.startswith("TwINFER"):
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.axvline(n_cols - 0.5, color="#888", lw=1.2)
    ax.axvline(n_cols + 0.5, color="#888", lw=1.2)
    ax.set_xlim(-0.5, n_cols_total - 0.5)
    ax.set_xticks(np.arange(-0.5, n_cols_total, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", bottom=False, left=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    return im


METRICS = [
    ("auprc", "AUPRC", 0.0, 1.0, "{:.2f}", "Score"),
    ("f1", "F1 (top-k, tie-aware)", 0.0, 1.0, "{:.2f}", "Score"),
    ("ratio_auprc", "AUPRC / random baseline", 0.0, 4.0, "{:.1f}x", "Ratio to random"),
    ("ratio_f1", "F1 (top-k) / random baseline", 0.0, 4.0, "{:.1f}x", "Ratio to random"),
]

DATASET_GROUPS = [
    ("Original real networks", ORIGINAL_DATASETS),
    ("Multistate & seeded (8 datasets)", MULTISTATE_DATASETS),
]

SIGN_MODES = [
    ("unsigned", BEELINE + TWINFER_UNSIGNED),
    ("signed (edge sign must match ground truth; sign from gene-gene correlation / edge weight)",
     BEELINE_SIGNED + TWINFER_SIGNED),
]


def main():
    df = pd.read_csv(REAL_NET_ROOT / "real_networks_all_nofilter_combined_scores.csv")
    out_path = REAL_NET_ROOT / "real_networks_all_nofilter_heatmap.pdf"

    with PdfPages(out_path) as pdf:
        for group_name, columns in DATASET_GROUPS:
            for sign_name, methods in SIGN_MODES:
                for metric, mlabel, vmin, vmax, fmt, cbar_label in METRICS:
                    sub = df[df.algorithm.isin(methods) & df.dataset.isin(columns)]
                    sub = sub[["dataset", "algorithm", metric]]
                    if sub.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(10.5, 6.5))
                    title = f"{mlabel}  --  {group_name}  --  {sign_name}"
                    im = make_panel(ax, sub, metric, columns, title, vmin=vmin, vmax=vmax, fmt=fmt)
                    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                    cbar.set_label(cbar_label, fontsize=9)
                    if metric.startswith("ratio"):
                        cbar.ax.axhline(1.0, color="red", lw=1.2)
                        cbar.ax.text(2.4, 1.0, "1x = random", fontsize=6.5, color="red", va="center", ha="left")
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
