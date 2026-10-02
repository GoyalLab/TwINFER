"""
Figure-3-style RAW correlation box plots (companion to
plot_figure_3_1k_zscores.py) for the causal_direction_inference conditions
(gene_1-gene_2). Same 3-panel layout and threshold-color rule, but plots
the underlying correlation coefficients instead of their z-scores:

  rho_correlation : gene_gene_matrix_results.csv gene_1_gene_2  (Step 1 raw rho, t1)
  rho_het         : twin_t1_matrix_results.csv gene_1_gene_2    (Step 2 raw twin rho_Delta, t1)
  rho_dagger      : box_plot_data.csv gene_1_to_gene_2 (solid, 1->2)
                     / gene_2_to_gene_1 (dashed, 2->1)           (Step 4 raw cross-rho)

NOT directional_matrix_results.csv for rho_dagger -- that's the GATED
pipeline's output and is exactly 0.0 for at least one condition
(A_and_B_both_repress, where Step 4 gets skipped by gating), while the
true forced value there is ~-0.30. box_plot_data.csv's gene_1_to_gene_2/
gene_2_to_gene_1 are the forced/unconditional reproduction, always
computed, same as the forced_* z-score columns used to color these boxes.

Box/median/whisker color still follows the corresponding Z-SCORE's
significance call (from zscores.csv, merged in by condition+rep_id) --
NOT a raw-correlation threshold -- so this plot directly shows "here's the
raw effect size for calls the z-score test made" and makes clear that raw
correlation magnitude alone does not track z-score significance (see the
zero-inflation / clonal non-independence discussion in the 2026-09-18
session: naive correlation magnitude is not a valid significance proxy on
this data).

Output: figure_3_1k_correlations.png/svg/pdf
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DEFAULT_AGG_DIR = (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/causal_direction_inference/figure_3_1k_run/results/aggregated')

_ap = argparse.ArgumentParser()
_ap.add_argument("agg_dir", nargs="?", default=DEFAULT_AGG_DIR,
                  help="aggregated/ dir containing zscores.csv, gene_gene_matrix_results.csv, "
                       "twin_t1_matrix_results.csv, directional_matrix_results.csv")
_ap.add_argument("out_dir", nargs="?", default=None,
                  help="output dir (default: figure_3_1k_run/plots next to agg_dir's run dir)")
_ap.add_argument("--threshold", type=float, default=2.33,
                  help="|z| significance line used for the box color rule (color still driven by z, not rho)")
_args = _ap.parse_args()

AGG_DIR = _args.agg_dir
OUT_DIR = _args.out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(AGG_DIR))), "plots")
os.makedirs(OUT_DIR, exist_ok=True)
THRESH = _args.threshold

# ============================================================
# Style (matches plot.ipynb / plot_v2.ipynb)
# ============================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ["Arial"]
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['svg.fonttype'] = "none"
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

try:
    import matplotlib.font_manager as fm
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "package"))
    for fname in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"]:
        fp = os.path.join(repo_root, "twinfer", "plotting", "assets", "fonts", fname)
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
except Exception as e:
    print("font load skipped:", e)

COND = [
    ("A_B",                  "A   B"),
    ("A_to_B",               "A --> B"),
    ("A_rep_B",              "A --| B"),
    ("A_and_B_both_repress", "A --| B\nB --| A"),
    ("A_rep_B_B_to_A",       "A --| B\nB --> A"),
    ("A_and_B",              "A --> B\nB --> A"),
]

z = pd.read_csv(os.path.join(AGG_DIR, "zscores.csv"))
gg = pd.read_csv(os.path.join(AGG_DIR, "gene_gene_matrix_results.csv"))
tw = pd.read_csv(os.path.join(AGG_DIR, "twin_t1_matrix_results.csv"))
box = pd.read_csv(os.path.join(AGG_DIR, "box_plot_data.csv")).rename(columns={"Condition": "condition"})

key = ["condition", "rep_id"]
df = z[key + ["forced_step1_z_t1", "forced_step2_z_het_t1", "forced_step4_z_1to2", "forced_step4_z_2to1"]].copy()
df = df.merge(gg[key + ["gene_1_gene_2"]].rename(columns={"gene_1_gene_2": "rho_correlation"}), on=key)
df = df.merge(tw[key + ["gene_1_gene_2"]].rename(columns={"gene_1_gene_2": "rho_het"}), on=key)
df = df.merge(box[key + ["gene_1_to_gene_2", "gene_2_to_gene_1"]].rename(
    columns={"gene_1_to_gene_2": "rho_dagger_1to2", "gene_2_to_gene_1": "rho_dagger_2to1"}), on=key)

COND = [c for c in COND if (df["condition"] == c[0]).any()]
df = df[df["condition"].isin({c[0] for c in COND})].copy()
conditions = [c[0] for c in COND]
n_by_cond = {c: int((df["condition"] == c).sum()) for c in conditions}
labels = [f"{lbl}\n(n={n_by_cond[c]})" for c, lbl in COND]


def get_threshold_color(z_values):
    """Color follows the Z-SCORE significance call, not the raw rho magnitude."""
    median_z = np.nanmedian(z_values)
    if median_z > THRESH:
        return "blue"
    elif median_z < -THRESH:
        return "red"
    return "grey"


def single_panel(ax, rho_col, z_col, ylabel):
    rho_by_cond = [df.loc[df["condition"] == c, rho_col].replace([np.inf, -np.inf], np.nan).dropna().values
                   for c in conditions]
    z_by_cond = [df.loc[df["condition"] == c, z_col].replace([np.inf, -np.inf], np.nan).dropna().values
                 for c in conditions]
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=1)
    box = ax.boxplot(rho_by_cond, positions=range(1, len(conditions) + 1),
                      patch_artist=True, showfliers=False, widths=0.5)
    for patch, zv in zip(box["boxes"], z_by_cond):
        color = get_threshold_color(zv)
        patch.set_facecolor("none"); patch.set_edgecolor(color); patch.set_linewidth(2); patch.set_clip_on(False)
    for median, zv in zip(box["medians"], z_by_cond):
        median.set_color(get_threshold_color(zv)); median.set_linewidth(2); median.set_clip_on(False)
    for part in ("whiskers", "caps"):
        for i, artist in enumerate(box[part]):
            idx = i // 2
            artist.set_color(get_threshold_color(z_by_cond[idx])); artist.set_linewidth(2); artist.set_clip_on(False)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def dagger_panel(ax, rho_col_12, rho_col_21, z_col_12, z_col_21, ylabel):
    rho_12 = [df.loc[df["condition"] == c, rho_col_12].replace([np.inf, -np.inf], np.nan).dropna().values
              for c in conditions]
    rho_21 = [df.loc[df["condition"] == c, rho_col_21].replace([np.inf, -np.inf], np.nan).dropna().values
              for c in conditions]
    z_12 = [df.loc[df["condition"] == c, z_col_12].replace([np.inf, -np.inf], np.nan).dropna().values
            for c in conditions]
    z_21 = [df.loc[df["condition"] == c, z_col_21].replace([np.inf, -np.inf], np.nan).dropna().values
            for c in conditions]
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=1)

    positions_12 = [i - 0.2 for i in range(1, len(conditions) + 1)]
    box_12 = ax.boxplot(rho_12, positions=positions_12, patch_artist=True, showfliers=False, widths=0.3)
    positions_21 = [i + 0.2 for i in range(1, len(conditions) + 1)]
    box_21 = ax.boxplot(rho_21, positions=positions_21, patch_artist=True, showfliers=False, widths=0.3)

    solid = mlines.Line2D([], [], color="black", linestyle="-",
                           label=r"$\hat{\rho}^{\dagger}_{1 \to 2}$")
    dashed = mlines.Line2D([], [], color="black", linestyle="--",
                            label=r"$\hat{\rho}^{\dagger}_{2 \to 1}$")
    ax.legend(handles=[solid, dashed], frameon=False)

    for patch, zv in zip(box_12["boxes"], z_12):
        color = get_threshold_color(zv)
        patch.set_facecolor("none"); patch.set_edgecolor(color); patch.set_linewidth(2); patch.set_clip_on(False)
    for median, zv in zip(box_12["medians"], z_12):
        median.set_color(get_threshold_color(zv)); median.set_linewidth(2); median.set_clip_on(False)

    for patch, zv in zip(box_21["boxes"], z_21):
        color = get_threshold_color(zv)
        patch.set_facecolor("none"); patch.set_edgecolor(color); patch.set_linestyle("--")
        patch.set_linewidth(2); patch.set_clip_on(False)
    for median, zv in zip(box_21["medians"], z_21):
        median.set_color(get_threshold_color(zv)); median.set_linewidth(2); median.set_clip_on(False)

    for box, z_list in [(box_12, z_12), (box_21, z_21)]:
        for i, whisker in enumerate(box["whiskers"]):
            idx = i // 2
            whisker.set_color(get_threshold_color(z_list[idx])); whisker.set_linewidth(2); whisker.set_clip_on(False)
        for i, cap in enumerate(box["caps"]):
            idx = i // 2
            cap.set_color(get_threshold_color(z_list[idx])); cap.set_linewidth(2); cap.set_clip_on(False)

    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(26, 7))
fig.patch.set_facecolor("none")
for ax in axes:
    ax.set_facecolor("none")

single_panel(axes[0], "rho_correlation", "forced_step1_z_t1", r"$\rho_{\rm correlation}$  (Step 1, $t_1$)")
single_panel(axes[1], "rho_het", "forced_step2_z_het_t1", r"$\rho_{\rm het}$  (Step 2, twin $\hat{\rho}_\Delta$, $t_1$)")
dagger_panel(axes[2], "rho_dagger_1to2", "rho_dagger_2to1",
             "forced_step4_z_1to2", "forced_step4_z_2to1", r"$\hat{\rho}^{\dagger}$  (Step 4)")

fig.suptitle(r"TwINFER raw correlations $\cdot$ gene_1-gene_2 $\cdot$ "
             f"{len(conditions)} conditions  (box color = z-score significance call, n per box in x-labels)",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

for ext in ("png", "svg", "pdf"):
    kwargs = dict(bbox_inches="tight", facecolor="none", edgecolor="none")
    if ext == "png":
        kwargs["dpi"] = 300
    else:
        kwargs["transparent"] = True
    fig.savefig(os.path.join(OUT_DIR, f"figure_3_1k_correlations.{ext}"), format=ext, **kwargs)
print("saved", os.path.join(OUT_DIR, "figure_3_1k_correlations.{png,svg,pdf}"))
plt.show()
