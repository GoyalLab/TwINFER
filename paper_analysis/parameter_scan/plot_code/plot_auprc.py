"""Precision-recall curves (AUPRC) of z_rho, z_rho_s and z_rho_Delta in the parameter scan.

Figure   : parameter_scan (used by heterogeneity_vs_regulation and causal_direction_inference).
Produces : z_rho_auprc.pdf, z_rho_s_auprc.pdf, z_rho_delta_auprc.pdf (precision against recall, exact square panels; one curve per topology in
           the colours and line styles of the ROC plots).
           The legend gives each curve's AUPRC and, in brackets, the AUPRC divided by its base rate (1.0 = chance).
Input    : plot_data/auprc_curves.csv (score, curve, recall, precision) and plot_data/auprc_summary.csv, made by
           analysis_code/auprc_of_z_scores.py, which also defines the positive and negative classes. No analysis is done here.
Style    : Illustrator-editable (Arial, cm math, fonttype 42, no clips/masks, transparent PDF).
Usage    : python plot_auprc.py [--plot-data-dir DIR] [--out-dir DIR]
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
                     "svg.fonttype": "none", "mathtext.fontset": "cm", "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})

# colours and line styles of the ROC plots (plot_roc_panels.py): the set for curves with multiple-state (two-state) systems among the positives
COLOURS = {"A_B": "#ee9127", "A_to_B": "#006937", "A_and_B": "#4292c6", "A_rep_B": "#7a5ca8"}
LINES = {"A_B": ":", "A_to_B": "-.", "A_and_B": "-", "A_rep_B": "--"}
NAMES = {"A_B": r"$\mathrm{X}\ \mathrm{Y}$", "A_to_B": r"$\mathrm{X} \rightarrow \mathrm{Y}$",
         "A_and_B": r"$\mathrm{X} \leftrightarrow \mathrm{Y}$", "A_rep_B": r"$\mathrm{X} \dashv \mathrm{Y}$"}
PLOTS = [("z_rho", "z_rho_auprc.pdf"), ("z_rho_s", "z_rho_s_auprc.pdf"), ("z_rho_Delta", "z_rho_delta_auprc.pdf")]

project = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")
ap = argparse.ArgumentParser()
ap.add_argument("--plot-data-dir", default=os.path.join(project, "final_figure_data", "parameter_scan", "plot_data"))
ap.add_argument("--out-dir", default=os.path.join(project, "final_plots", "parameter_scan"))
args = ap.parse_args()

# [2026-10-24] z_rho_delta, q_rho_delta, rho_delta renamed to z_rho_Delta, q_rho_Delta, rho_Delta: csvs written before the rename are read with the new names
curves = pd.read_csv(os.path.join(args.plot_data_dir, "auprc_curves.csv"))
curves["score"] = curves["score"].str.replace("rho_delta", "rho_Delta")
# [2026-10-24 replaced: score values renamed before set_index]
# summary = pd.read_csv(os.path.join(args.plot_data_dir, "auprc_summary.csv")).set_index(["score", "curve"])
summary = pd.read_csv(os.path.join(args.plot_data_dir, "auprc_summary.csv"))
summary["score"] = summary["score"].str.replace("rho_delta", "rho_Delta")
summary = summary.set_index(["score", "curve"])

for score, pdf_name in PLOTS:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    for curve in COLOURS:
        if (score, curve) not in summary.index:
            continue
        row, points = summary.loc[(score, curve)], curves[(curves.score == score) & (curves.curve == curve)]
        ax.plot(points.recall, points.precision, LINES[curve], color=COLOURS[curve], linewidth=1.5,
                label=f"{NAMES[curve]}  {row.AUPRC:.3f} (x{row.AUPRC_over_base:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)                                # x and y both run 0-1: exact square, as in the ROC plots
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    ax.legend(frameon=False, loc="lower left", title="AUPRC (x base rate)", title_fontsize=10, handlelength=2.6)
    for artist in fig.findobj():                  # no clips/masks (Illustrator)
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)
    fig.savefig(os.path.join(args.out_dir, pdf_name), format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    plt.close(fig)
    print(pdf_name)
