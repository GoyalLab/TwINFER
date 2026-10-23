"""Effect of the number of twin pairs on z_rho, z_rho_s and z_rho_delta for the six heterogeneity_vs_regulation scenarios.

Figure   : heterogeneity_vs_regulation (A,B vs A->B; single state, 2 states, drift).
Produces : z_rho_effect_n_twin.pdf, z_rho_s_effect_n_twin.pdf, z_rho_delta_effect_n_twin.pdf
           (x = scenario, within each scenario one box per number of twin pairs, 100 replicates; shaded greyscale, light = few, dark = many).
Input    : analysis_data/twin_covariance_decomposition_z_scores.csv (made by analysis_code/twin_covariance_decomposition_z_scores.py);
           columns scenario, rep, N, z_rho, z_rho_s, z_rho_delta. Other values of N in the csv are ignored. No analysis is done here.
Y axes   : z_rho and z_rho_s share the same y-axis limits.
Style    : Illustrator-editable (Arial, cm math, fonttype 42, no clips/masks, transparent PDF); filled grey boxes with black edges,
           outliers shown, light y grid, dotted zero line, dashed lines at +-1.96 (same format as the earlier by-network z_rho_delta figure).

Usage    : python plot_n_twins_effect.py [--data CSV] [--out-dir DIR]
"""
import argparse
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

matplotlib.use("Agg")                                   # draw to files only, no display needed
FONT_DIR = str(pathlib.Path(__file__).resolve().parents[3] / "package" / "twinfer" / "plotting" / "assets" / "fonts") + "/"   # Arial shipped with the package
for font_file in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]:
    fm.fontManager.addfont(FONT_DIR + font_file)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.sans-serif": ["Arial"], "font.family": "sans-serif",
                     "svg.fonttype": "none", "mathtext.fontset": "cm", "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12, "legend.title_fontsize": 12})

N_TWINS = [250, 500, 1000, 3000, 6000]
GREY = ["#b8b8b8", "#9a9a9a", "#7c7c7c", "#4f4f4f", "#1a1a1a"]            # box fill, light = few twin pairs
MEDIAN = ["#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff", "#ffffff"]          # median line: black on light, white on dark boxes
THRESHOLD = 1.96                                                           # dashed lines at +-THRESHOLD
SCENARIOS = ["A_B (single state)", "A_to_B (single state)", "A_B (two states)", "A_to_B (two states)", "A_B (drift)", "A_to_B (drift)"]
LABELS = [r"$\mathrm{A,\ B}$", r"$\mathrm{A} \rightarrow \mathrm{B}$", r"$\mathrm{A,\ B}$" "\n2 states", r"$\mathrm{A} \rightarrow \mathrm{B}$" "\n2 states",
          r"$\mathrm{A,\ B}$" "\ndrift", r"$\mathrm{A} \rightarrow \mathrm{B}$" "\ndrift"]            # upright math text (Computer Modern), second line in Arial
PLOTS = [("z_rho", r"$z$-score of expression correlation $z_\rho$", "z_rho_effect_n_twin.pdf"),
         ("z_rho_s", r"$z$-score of twin-shared correlation $z_{\rho_s}$", "z_rho_s_effect_n_twin.pdf"),
         ("z_rho_delta", r"$z$-score of twin-difference correlation $z_{\hat{\rho}_\Delta}$", "z_rho_delta_effect_n_twin.pdf")]
BOX_SPACING = 0.16                                                         # distance between the boxes of one scenario

figure_data = os.path.join(os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA"), "final_figure_data", "heterogeneity_vs_regulation")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(figure_data, "analysis_data", "twin_covariance_decomposition_z_scores.csv"))
ap.add_argument("--out-dir", default=os.path.join(os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA"), "final_plots", "heterogeneity_vs_regulation"))
args = ap.parse_args()

data = pd.read_csv(args.data)

SHARED_YLIM = ("z_rho", "z_rho_s")                  # statistics drawn with the same y-axis limits
figures = []
for column, ylabel, pdf_name in PLOTS:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    for i, scenario in enumerate(SCENARIOS):
        for j, n in enumerate(N_TWINS):
            values = data.loc[(data.scenario == scenario) & (data.N == n), column].dropna().values
            position = i + (j - (len(N_TWINS) - 1) / 2) * BOX_SPACING
            box = ax.boxplot(values, positions=[position], widths=BOX_SPACING * 0.88, patch_artist=True, showfliers=True,
                             medianprops=dict(color=MEDIAN[j], lw=1.4), whiskerprops=dict(color="#0b0b0b", lw=0.9),
                             capprops=dict(color="#0b0b0b", lw=0.9), boxprops=dict(color="#0b0b0b", lw=0.8),
                             flierprops=dict(marker="o", ms=2.4, mfc="none", mec="#52514e", mew=0.7))
            box["boxes"][0].set_facecolor(GREY[j])
    for y in (THRESHOLD, -THRESHOLD):
        ax.axhline(y, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1)
    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels(LABELS)
    ax.set_xlim(-0.55, len(SCENARIOS) - 0.45)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(handles=[Patch(fc=GREY[j], ec="#0b0b0b", lw=0.8, label=str(n)) for j, n in enumerate(N_TWINS)], title="number of twin pairs",
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(N_TWINS), frameon=False, columnspacing=1.4, handlelength=1.4)
    figures.append((column, fig, ax, pdf_name))

# z_rho and z_rho_s are drawn on the same y-axis limits (the union of the two automatic ranges)
shared = [ax.get_ylim() for column, fig, ax, pdf_name in figures if column in SHARED_YLIM]
for column, fig, ax, pdf_name in figures:
    if column in SHARED_YLIM:
        ax.set_ylim(min(lo for lo, hi in shared), max(hi for lo, hi in shared))
    for artist in fig.findobj():                  # no clips/masks (Illustrator)
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)
    fig.savefig(os.path.join(args.out_dir, pdf_name), format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    plt.close(fig)
    print(pdf_name)
