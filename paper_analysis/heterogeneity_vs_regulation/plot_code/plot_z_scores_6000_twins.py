"""Box plots of z_rho, z_rho_s and z_rho_delta for the six heterogeneity_vs_regulation scenarios with 6,000 twin pairs.

Figure   : heterogeneity_vs_regulation (A,B vs A->B; single state, 2 states, drift).
Produces : z_rho_6000_twins.pdf, z_rho_s_6000_twins.pdf, z_rho_delta_6000_twins.pdf  (one box per scenario, 100 replicates).
Input    : analysis_data/twin_covariance_decomposition_z_scores.csv (made by analysis_code/twin_covariance_decomposition_z_scores.py);
           rows with N == 6000 are used, columns scenario, rep, z_rho, z_rho_s, z_rho_delta. No analysis is done here.
Y axes   : z_rho and z_rho_s share the same y-axis limits.
Style    : Illustrator-editable (Arial, cm math, fonttype 42, no clips/masks, transparent PDF), unfilled boxes without outliers,
           dashed lines at +-1.96. Colours: the four fixed scenario colours of heterogeneity_vs_regulation/plot_v2.ipynb
           (A,B / A->B / 2-state A->B / 2-state A,B) plus blue and purple for the two drift scenarios.

Usage    : python plot_z_scores_6000_twins.py [--data CSV] [--out-dir DIR]
"""
import argparse
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm

matplotlib.use("Agg")                                   # draw to files only, no display needed
FONT_DIR = str(pathlib.Path(__file__).resolve().parents[3] / "package" / "twinfer" / "plotting" / "assets" / "fonts") + "/"   # Arial shipped with the package
for font_file in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]:
    fm.fontManager.addfont(FONT_DIR + font_file)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.sans-serif": ["Arial"], "font.family": "sans-serif",
                     "svg.fonttype": "none", "mathtext.fontset": "cm", "axes.labelsize": 18, "xtick.labelsize": 12, "ytick.labelsize": 12})

N_TWINS = 6000
THRESHOLD = 1.96                                  # dashed lines at +-THRESHOLD
SCENARIOS = ["A_B (single state)", "A_to_B (single state)", "A_B (two states)", "A_to_B (two states)", "A_B (drift)", "A_to_B (drift)"]
LABELS = [r"$\mathrm{A,\ B}$", r"$\mathrm{A} \rightarrow \mathrm{B}$", r"$\mathrm{A,\ B}$" "\n2 states", r"$\mathrm{A} \rightarrow \mathrm{B}$" "\n2 states",
          r"$\mathrm{A,\ B}$" "\ndrift", r"$\mathrm{A} \rightarrow \mathrm{B}$" "\ndrift"]            # upright math text (Computer Modern), second line in Arial
COLOURS = ["#bc2462", "#754c29", "#ee8d39", "#006838", "#2a6fb0", "#7b4fa3"]
PLOTS = [("z_rho", r"$z$-score of expression correlation $z_\rho$", "z_rho_6000_twins.pdf"),
         ("z_rho_s", r"$z$-score of twin-shared correlation $z_{\rho_s}$", "z_rho_s_6000_twins.pdf"),
         ("z_rho_delta", r"$z$-score of twin-difference correlation $z_{\hat{\rho}_\Delta}$", "z_rho_delta_6000_twins.pdf")]

figure_data = os.path.join(os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA"), "final_figure_data", "heterogeneity_vs_regulation")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(figure_data, "analysis_data", "twin_covariance_decomposition_z_scores.csv"))
ap.add_argument("--out-dir", default=os.path.join(os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA"), "final_plots", "heterogeneity_vs_regulation"))
args = ap.parse_args()

data = pd.read_csv(args.data)
data = data[data.N == N_TWINS]

SHARED_YLIM = ("z_rho", "z_rho_s")                  # statistics drawn with the same y-axis limits
figures = []
for column, ylabel, pdf_name in PLOTS:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    for y in (THRESHOLD, -THRESHOLD):
        ax.axhline(y, linestyle="--", color="black", linewidth=1, alpha=0.7, dashes=(12, 4))
    values = [data.loc[data.scenario == s, column].dropna().values for s in SCENARIOS]
    box = ax.boxplot(values, positions=range(1, len(SCENARIOS) + 1), patch_artist=True, showfliers=False, widths=0.3)
    for i, colour in enumerate(COLOURS):
        box["boxes"][i].set_facecolor("none")
        box["boxes"][i].set_edgecolor(colour)
        box["boxes"][i].set_linewidth(2)
        box["medians"][i].set_color(colour)
        box["medians"][i].set_linewidth(2)
        for line in box["whiskers"][2 * i:2 * i + 2] + box["caps"][2 * i:2 * i + 2]:
            line.set_color(colour)
            line.set_linewidth(2)
    ax.set_xticks(range(1, len(SCENARIOS) + 1))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
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
