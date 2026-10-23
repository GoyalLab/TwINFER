"""figure2_z_rho_delta_6000_pairs.pdf recoloured with plot_v2.ipynb's ORIGINAL scheme (create_box_plot_and_save): unfilled boxes, no fliers, linewidth 2, box/whisker/cap/median all one colour set by the group median vs the +-1.96 threshold: blue if median > 1.96, red if median < -1.96, grey otherwise; black dashed threshold lines.
Data: 6000 twin pairs, all clones, 100 replicates (twin_size_scan_figure2_6000.csv).  Illustrator-editable recipe, no title."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.sans-serif": ["Arial"], "font.family": "sans-serif", "svg.fonttype": "none", "mathtext.fontset": "cm", "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/"; OUT = A + "final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; PREVIEW = os.environ.get("PREVIEW_DIR"); TH = 1.96
D = pd.read_csv(A + "causal_direction_inference/zrho_pdf_network_auprc/twin_size_scan_figure2_6000.csv")
order = ["A_B (single state)", "A_to_B (single state)", "A_B (two states)", "A_to_B (two states)", "A_B (drift)", "A_to_B (drift)"]; lab = ["A, B", "A→B", "A, B\n2 states", "A→B\n2 states", "A, B\ndrift", "A→B\ndrift"]
col = lambda v: "blue" if np.median(v) > TH else ("red" if np.median(v) < -TH else "grey")
fig, ax = plt.subplots(figsize=(11, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0)
ax.axhline(TH, linestyle="--", color="black", linewidth=1, alpha=0.7, dashes=(12, 4)); ax.axhline(-TH, linestyle="--", color="black", linewidth=1, alpha=0.7, dashes=(12, 4))
vals = [D[D.scenario == sc].z_U.dropna().values for sc in order]
b = ax.boxplot(vals, positions=range(1, 7), patch_artist=True, showfliers=False, widths=0.3)
for i, v in enumerate(vals):
    c = col(v); b["boxes"][i].set_facecolor("none"); b["boxes"][i].set_edgecolor(c); b["boxes"][i].set_linewidth(2); b["medians"][i].set_color(c); b["medians"][i].set_linewidth(2)
    for w in b["whiskers"][2 * i:2 * i + 2] + b["caps"][2 * i:2 * i + 2]: w.set_color(c); w.set_linewidth(2)
ax.set_xticks(range(1, 7)); ax.set_xticklabels(lab); ax.set_ylabel(r"$z_{\hat\rho_\Delta}$"); ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
for artist in fig.findobj():
    if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
fig.savefig(OUT + "figure2_z_rho_delta_6000_pairs.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
if PREVIEW: fig.savefig(PREVIEW + "/figure2_zU_6000_orig_colours.png", dpi=110, bbox_inches="tight", facecolor="white")
print({sc: (round(float(np.median(v)), 2), col(v)) for sc, v in zip(order, vals)})
