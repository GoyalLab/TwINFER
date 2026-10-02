"""
Figure-3-style z-score box plots for the 6 causal_direction_inference
conditions (gene_1-gene_2), from zscores.csv (see the 2026-09-17 handoff
doc for column definitions). Style matches plot.ipynb's
create_box_plot_and_save (threshold-colored boxes, solid/dashed pair for
direction) rather than the 6-scenario grid layout. A_B and A_to_B are the
figure_2 baseline conditions (no-regulation / single-activation), added via
run_figure2_ab_inference_slurm.sh against the ~1000-rep
figure_2_simulations_1000 data; the other 4 come from figure_3_1k_run.
All 3 panels share one y-axis scale so magnitudes are directly comparable.

Panels (all "forced_*", t1 only -- matches the gated pipeline's own
routing, which only evaluates Step 1/2 at t1):
  z_correlation : forced_step1_z_t1        (Step 1 gene-gene correlation z)
  z_het         : forced_step2_z_het_t1    (Step 2 heterogeneity z)
  z_dagger      : forced_step4_z_1to2 (solid) / forced_step4_z_2to1 (dashed)
                  (Step 4 twin cross-correlation z, both directions)

Output: figure_3_1k_zscores.png/svg/pdf
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DEFAULT_CSV = (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/causal_direction_inference/figure_3_1k_run/results/aggregated/zscores.csv')

_ap = argparse.ArgumentParser()
_ap.add_argument("csv", nargs="?", default=DEFAULT_CSV)
_ap.add_argument("out_dir", nargs="?", default=None,
                  help="output dir (default: figure_3_1k_run/plots next to the input CSV's run dir)")
_ap.add_argument("--threshold", type=float, default=2.33,
                  help="|z| significance line, also used for the box color rule")
_args = _ap.parse_args()

CSV = _args.csv
OUT_DIR = _args.out_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(CSV)))), "plots")
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

df = pd.read_csv(CSV)
COND = [c for c in COND if (df["condition"] == c[0]).any()]
df = df[df["condition"].isin({c[0] for c in COND})].copy()
conditions = [c[0] for c in COND]
n_by_cond = {c: int((df["condition"] == c).sum()) for c in conditions}
labels = [f"{lbl}\n(n={n_by_cond[c]})" for c, lbl in COND]

ALL_Z_COLS = ["forced_step1_z_t1", "forced_step2_z_het_t1", "forced_step4_z_1to2", "forced_step4_z_2to1"]


def shared_ylim(pad=0.06):
    v = pd.concat([df[c] for c in ALL_Z_COLS]).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = v.min(), v.max()
    m = (hi - lo) * pad or 0.1
    return lo - m, hi + m


YLIM = shared_ylim()


def get_threshold_color(values):
    """Box color rule from create_box_plot_and_save: blue if the median
    clears +THRESH, red if it clears -THRESH, grey (ambiguous) otherwise."""
    median_val = np.nanmedian(values)
    if median_val > THRESH:
        return "blue"
    elif median_val < -THRESH:
        return "red"
    return "grey"


def single_panel(ax, col, ylabel):
    values_by_cond = [df.loc[df["condition"] == c, col].replace([np.inf, -np.inf], np.nan).dropna().values
                       for c in conditions]
    ax.axhline(THRESH, linestyle="--", color="black", linewidth=1, alpha=0.7, dashes=(12, 4))
    ax.axhline(-THRESH, linestyle="--", color="black", linewidth=1, alpha=0.7, dashes=(12, 4))
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=1)
    box = ax.boxplot(values_by_cond, positions=range(1, len(conditions) + 1),
                      patch_artist=True, showfliers=False, widths=0.5)
    for patch, values in zip(box["boxes"], values_by_cond):
        color = get_threshold_color(values)
        patch.set_facecolor("none"); patch.set_edgecolor(color); patch.set_linewidth(2); patch.set_clip_on(False)
    for median, values in zip(box["medians"], values_by_cond):
        median.set_color(get_threshold_color(values)); median.set_linewidth(2); median.set_clip_on(False)
    for part in ("whiskers", "caps"):
        for i, artist in enumerate(box[part]):
            idx = i // 2
            color = get_threshold_color(values_by_cond[idx])
            artist.set_color(color); artist.set_linewidth(2); artist.set_clip_on(False)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*YLIM)
    ax.spines[["top", "right"]].set_visible(False)


def dagger_panel(ax, col_12, col_21, ylabel):
    values_12 = [df.loc[df["condition"] == c, col_12].replace([np.inf, -np.inf], np.nan).dropna().values
                 for c in conditions]
    values_21 = [df.loc[df["condition"] == c, col_21].replace([np.inf, -np.inf], np.nan).dropna().values
                 for c in conditions]
    ax.axhline(THRESH, linestyle="--", color="black", linewidth=1, alpha=0.7, dashes=(12, 4))
    ax.axhline(-THRESH, linestyle="--", color="black", linewidth=1, alpha=0.7, dashes=(12, 4))
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=1)

    positions_12 = [i - 0.2 for i in range(1, len(conditions) + 1)]
    box_12 = ax.boxplot(values_12, positions=positions_12, patch_artist=True, showfliers=False, widths=0.3)
    positions_21 = [i + 0.2 for i in range(1, len(conditions) + 1)]
    box_21 = ax.boxplot(values_21, positions=positions_21, patch_artist=True, showfliers=False, widths=0.3)

    solid = mlines.Line2D([], [], color="black", linestyle="-",
                           label=r"$z(\hat{\rho}^{\dagger}_{1 \to 2})$")
    dashed = mlines.Line2D([], [], color="black", linestyle="--",
                            label=r"$z(\hat{\rho}^{\dagger}_{2 \to 1})$")
    ax.legend(handles=[solid, dashed], frameon=False)

    for patch, values in zip(box_12["boxes"], values_12):
        color = get_threshold_color(values)
        patch.set_facecolor("none"); patch.set_edgecolor(color); patch.set_linewidth(2); patch.set_clip_on(False)
    for median, values in zip(box_12["medians"], values_12):
        median.set_color(get_threshold_color(values)); median.set_linewidth(2); median.set_clip_on(False)

    for patch, values in zip(box_21["boxes"], values_21):
        color = get_threshold_color(values)
        patch.set_facecolor("none"); patch.set_edgecolor(color); patch.set_linestyle("--")
        patch.set_linewidth(2); patch.set_clip_on(False)
    for median, values in zip(box_21["medians"], values_21):
        median.set_color(get_threshold_color(values)); median.set_linewidth(2); median.set_clip_on(False)

    for box, values_list in [(box_12, values_12), (box_21, values_21)]:
        for i, whisker in enumerate(box["whiskers"]):
            idx = i // 2
            color = get_threshold_color(values_list[idx])
            whisker.set_color(color); whisker.set_linewidth(2); whisker.set_clip_on(False)
        for i, cap in enumerate(box["caps"]):
            idx = i // 2
            color = get_threshold_color(values_list[idx])
            cap.set_color(color); cap.set_linewidth(2); cap.set_clip_on(False)

    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*YLIM)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(26, 7))
fig.patch.set_facecolor("none")
for ax in axes:
    ax.set_facecolor("none")

single_panel(axes[0], "forced_step1_z_t1", r"$z_{\rm correlation}$  (Step 1, $t_1$)")
single_panel(axes[1], "forced_step2_z_het_t1", r"$z_{\rm het}$  (Step 2, $t_1$)")
dagger_panel(axes[2], "forced_step4_z_1to2", "forced_step4_z_2to1", r"$z_{\dagger}$  (Step 4)")

fig.suptitle(r"TwINFER z-scores $\cdot$ gene_1-gene_2 $\cdot$ "
             f"{len(conditions)} conditions (n per box in x-labels)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])

for ext in ("png", "svg", "pdf"):
    kwargs = dict(bbox_inches="tight", facecolor="none", edgecolor="none")
    if ext == "png":
        kwargs["dpi"] = 300
    else:
        kwargs["transparent"] = True
    fig.savefig(os.path.join(OUT_DIR, f"figure_3_1k_zscores.{ext}"), format=ext, **kwargs)
print("saved", os.path.join(OUT_DIR, "figure_3_1k_zscores.{png,svg,pdf}"))
plt.show()
