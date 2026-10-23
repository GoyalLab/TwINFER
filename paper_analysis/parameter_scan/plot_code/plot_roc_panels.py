"""ROC panels of the parameter scan: scenario / multiple-state detection and regulation among two-state systems.

Figure   : parameter_scan (used by heterogeneity_vs_regulation and causal_direction_inference).
Produces : roc_panels.pdf (three square panels)
             regulation (true positive rate) against scenario 1 (false positive rate), score z_rho; X <-> Y, X -> Y, X -| Y single-state systems
             multiple states (TPR) against scenario 1 (FPR), score z_rho; the two-state systems of the four topologies
             multiple states (TPR) against single state (FPR), score z_rho_s; twin pairs (solid) and the random-pair control (dashed)
           roc_two_states_regulation.pdf (one square panel)
             two states with regulation (TPR) against two states without regulation (FPR), score z_rho_delta; X <-> Y, X -> Y, X -| Y
           Scenario 1 = single-state A_B (no regulation, single state). Each panel gives the AUROC of its curves.
Input    : plot_data/roc_curves.csv and plot_data/roc_summary.csv, made by analysis_code/roc_of_z_scores.py, which also defines the
           classes. No analysis is done here.
Style    : Illustrator-editable (Arial, cm math, fonttype 42, no clips/masks, transparent PDF), exact square panels (both axes 0-1). Single-state curves and
           multiple-state curves use two different colour sets; a topology keeps its colour in all panels of its state type. The script stops with an error if a panel is not square or if a
           text or legend overlaps the diagonal or another text or legend.
Usage    : python plot_roc_panels.py [--plot-data-dir DIR] [--out-dir DIR]
Requires : matplotlib, pandas (the twinfer-code environment); the Arial fonts shipped in package/twinfer/plotting/assets/fonts.
"""
import argparse
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

matplotlib.use("Agg")                                   # draw to files only, no display needed
FONT_DIR = str(pathlib.Path(__file__).resolve().parents[3] / "package" / "twinfer" / "plotting" / "assets" / "fonts") + "/"   # Arial shipped with the package
for font_file in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]:
    fm.fontManager.addfont(FONT_DIR + font_file)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.sans-serif": ["Arial"], "font.family": "sans-serif", "svg.fonttype": "none",
                     "mathtext.fontset": "cm", "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})

NAMES = {"A_B": r"$\mathrm{X}\ \mathrm{Y}$", "A_to_B": r"$\mathrm{X} \rightarrow \mathrm{Y}$",
         "A_and_B": r"$\mathrm{X} \leftrightarrow \mathrm{Y}$", "A_rep_B": r"$\mathrm{X} \dashv \mathrm{Y}$"}
# colours of the earlier ROC figure (parameter_space_scan/plot.ipynb, fixed_colors): one set for curves whose positives are single-state systems and a
# different set for curves whose positives are multiple-state (two-state) systems; each topology keeps its colour in all panels of its state type.
# The line style belongs to the topology (the random-pair control is dashed, so the panel with that control draws its twin curves solid).
COLOURS_SINGLE = {"A_B": "#bc0f58", "A_to_B": "#6b4b2e", "A_and_B": "#084594", "A_rep_B": "#800026"}
COLOURS_TWO_STATES = {"A_B": "#ee9127", "A_to_B": "#006937", "A_and_B": "#4292c6", "A_rep_B": "#7a5ca8"}
LINES = {"A_B": ":", "A_to_B": "-.", "A_and_B": "-", "A_rep_B": "--"}
styles_of = lambda order, colours, solid=False: {c: (colours[c], "-" if solid else LINES[c]) for c in order}
# everything that differs between panels: curves in order, (colour, style), axis labels, where the AUROC lines start (axes coordinates)
PANELS = {
    "regulation_vs_scenario_1": dict(order=["A_and_B", "A_to_B", "A_rep_B"], styles=styles_of(["A_and_B", "A_to_B", "A_rep_B"], COLOURS_SINGLE), auroc_at=(0.50, 0.40), auroc_names=False,
                                     labels=("Scenario 1 (false positive rate)", "regulation (true positive rate)"), random_guess=True),
    "multiple_states_vs_scenario_1": dict(order=["A_and_B", "A_to_B", "A_rep_B", "A_B"],
                                          styles=styles_of(["A_and_B", "A_to_B", "A_rep_B", "A_B"], COLOURS_TWO_STATES),
                                          auroc_at=(0.04, 0.70), auroc_names=False, step=0.075,
                                          labels=("Scenario 1 (false positive rate)", "multiple states (true positive rate)")),
    "multiple_states_vs_single_state": dict(order=["A_B", "A_and_B", "A_to_B", "A_rep_B"],
                                            styles=styles_of(["A_B", "A_and_B", "A_to_B", "A_rep_B"], COLOURS_TWO_STATES, solid=True),
                                            auroc_at=(0.50, 0.40), auroc_names=True, twins_legend=True,
                                            labels=("single state (false positive rate)", "multiple states (true positive rate)")),
    "regulation_two_states": dict(order=["A_and_B", "A_to_B", "A_rep_B"], styles=styles_of(["A_and_B", "A_to_B", "A_rep_B"], COLOURS_TWO_STATES), auroc_at=(0.50, 0.40), auroc_names=False,
                                  labels=("two states, no regulation (false positive rate)", "two states, regulation (true positive rate)"), random_guess=True),
}
FIGURES = {"roc_panels.pdf": (["regulation_vs_scenario_1", "multiple_states_vs_scenario_1", "multiple_states_vs_single_state"], (14, 4.8)),
           "roc_two_states_regulation.pdf": (["regulation_two_states"], (5.2, 4.8))}

project = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")
ap = argparse.ArgumentParser()
ap.add_argument("--plot-data-dir", default=os.path.join(project, "final_figure_data", "parameter_scan", "plot_data"))
ap.add_argument("--out-dir", default=os.path.join(project, "final_plots", "parameter_scan"))
args = ap.parse_args()
curves = pd.read_csv(os.path.join(args.plot_data_dir, "roc_curves.csv"))
summary = pd.read_csv(os.path.join(args.plot_data_dir, "roc_summary.csv"))


def draw(ax, panel, texts):
    """One square ROC panel; texts collects every text and legend that must stay clear of the diagonal and of each other."""
    cfg = PANELS[panel]
    ax.set_facecolor("none")
    ax.plot([0, 1], [0, 1], color="#b0b0b0", linewidth=1.5)
    for curve in cfg["order"]:
        colour, style = cfg["styles"][curve]
        for source in ("twins", "random_pairs"):
            points = curves[(curves.panel == panel) & (curves.curve == curve) & (curves.source == source)]
            if not points.empty:
                ax.plot(points.fpr, points.tpr, style if source == "twins" else "--", color=colour, linewidth=1.5 if source == "twins" else 1.0)
    auroc = {curve: summary[(summary.panel == panel) & (summary.curve == curve) & (summary.source == "twins")].AUROC.iloc[0] for curve in cfg["order"]}
    for k, curve in enumerate(sorted(cfg["order"], key=lambda c: -auroc[c])):   # AUROC of the twin-pair curves, one coloured line per curve, highest on top
        value = auroc[curve]
        text = f"{NAMES[curve]}  AUROC = {value:.3f}" if cfg["auroc_names"] else f"AUROC = {value:.3f}"
        x, y = cfg["auroc_at"][0], cfg["auroc_at"][1] - cfg.get("step", 0.065) * k
        texts.append(ax.text(x, y, text, color=cfg["styles"][curve][0], fontsize=11, fontweight="bold", transform=ax.transAxes))
    if cfg.get("twins_legend"):
        handles = [Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="twins"), Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="random pairs")]
    else:
        handles = [Line2D([0], [0], color=cfg["styles"][c][0], linestyle=cfg["styles"][c][1], linewidth=1.5, label=NAMES[c]) for c in cfg["order"]]
    texts.append(ax.legend(handles=handles, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=2.6))
    if cfg.get("random_guess"):
        texts.append(ax.text(0.78, 0.58, "random\nguess", color="#b0b0b0", fontsize=10, ha="left", transform=ax.transAxes))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)                                                        # x and y both run 0-1: every panel is an exact square
    ax.set_xlabel(cfg["labels"][0])
    ax.set_ylabel(cfg["labels"][1])
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def check(fig, axes, texts):
    """Stop if a panel is not square, or a text / legend overlaps the diagonal or another text / legend of its panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        box = ax.get_window_extent(renderer)
        if abs(box.width - box.height) > 0.5:
            raise SystemExit(f"a panel is not square: {box.width:.1f} x {box.height:.1f} px")
    boxes = [(item, item.get_window_extent(renderer)) for item in texts]
    for n, (item, box) in enumerate(boxes):
        for other, other_box in boxes[n + 1:]:
            if item.axes is other.axes and box.overlaps(other_box):
                raise SystemExit("two texts or legends overlap in a panel")
    for item, box in boxes:                                                     # the diagonal crosses a box [x0,x1] x [y0,y1] (axes coordinates) when the ranges meet
        (x0, y0), (x1, y1) = item.axes.transAxes.inverted().transform([[box.x0, box.y0], [box.x1, box.y1]])
        if max(x0, y0) <= min(x1, y1):
            raise SystemExit(f"a text or legend overlaps the diagonal at x {x0:.2f}-{x1:.2f}, y {y0:.2f}-{y1:.2f}")


for file_name, (panels, size) in FIGURES.items():
    fig, axes = plt.subplots(1, len(panels), figsize=size, squeeze=False)
    axes = list(axes[0])
    texts = []
    fig.patch.set_alpha(0)
    fig.subplots_adjust(wspace=0.34)
    for ax, panel in zip(axes, panels):
        draw(ax, panel, texts)
    check(fig, axes, texts)
    for artist in fig.findobj():                                                # no clips/masks (Illustrator)
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)
    fig.savefig(os.path.join(args.out_dir, file_name), format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    plt.close(fig)
    print(file_name)
