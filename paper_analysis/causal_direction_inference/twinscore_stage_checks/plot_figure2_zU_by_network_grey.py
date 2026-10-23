"""z_rho_hat_Delta (= z_U) for the figure 2 scenarios, grouped by network, with 250, 500, 1000, 3000 and 6000 twin pairs per group in a greyscale ramp (light = few pairs, dark = many).  Nested subsets of one clone permutation per replicate; 6000 = all clones.
Illustrator-editable recipe (Arial + cm math, guide font sizes, no clips, transparent PDF, no title/description on the plot).  Output: new_metrics_designs/figure2_by_network_z_rho_delta.pdf"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42; plt.rcParams["font.sans-serif"] = ["Arial"]; plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["svg.fonttype"] = "none"; plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.labelsize"] = 18; plt.rcParams["axes.titlesize"] = 20; plt.rcParams["xtick.labelsize"] = 12; plt.rcParams["ytick.labelsize"] = 12; plt.rcParams["legend.fontsize"] = 12; plt.rcParams["legend.title_fontsize"] = 12
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/"; O = A + "causal_direction_inference/zrho_pdf_network_auprc/"; OUT = A + "final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; PREVIEW = os.environ.get("PREVIEW_DIR")
NS = [250, 500, 1000, 3000, 6000]; GREY = ["#b8b8b8", "#9a9a9a", "#7c7c7c", "#4f4f4f", "#1a1a1a"]; MEDIAN = ["#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff", "#ffffff"]      # median line black on light, white on dark boxes
D = pd.concat([pd.read_csv(O + "twin_size_scan_figure2.csv").query("N in @NS"), pd.read_csv(O + "twin_size_scan_figure2_6000.csv")], ignore_index=True)
order = ["A_B (single state)", "A_to_B (single state)", "A_B (two states)", "A_to_B (two states)", "A_B (drift)", "A_to_B (drift)"]; labels = ["A, B", "A→B", "A, B\n2 states", "A→B\n2 states", "A, B\ndrift", "A→B\ndrift"]
fig, ax = plt.subplots(figsize=(13, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); w = 0.16; k = len(NS)
for i, sc in enumerate(order):
    for j, N in enumerate(NS):
        v = D[(D.scenario == sc) & (D.N == N)].z_U.dropna().values
        b = ax.boxplot(v, positions=[i + (j - (k - 1) / 2) * w], widths=w * 0.88, patch_artist=True, showfliers=True, medianprops=dict(color=MEDIAN[j], lw=1.4), whiskerprops=dict(color="#0b0b0b", lw=0.9), capprops=dict(color="#0b0b0b", lw=0.9),
                       boxprops=dict(color="#0b0b0b", lw=0.8), flierprops=dict(marker="o", ms=2.4, mfc="none", mec="#52514e", mew=0.7)); b["boxes"][0].set_facecolor(GREY[j])
for s_ in (1.96, -1.96): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(len(order))); ax.set_xticklabels(labels); ax.set_xlim(-0.55, len(order) - 0.45); ax.set_ylabel(r"$z_{\hat\rho_\Delta}$"); ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
ax.legend(handles=[Patch(fc=GREY[j], ec="#0b0b0b", lw=0.8, label=str(NS[j])) for j in range(k)], title="number of twin pairs", loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, columnspacing=1.4, handlelength=1.4)
for artist in fig.findobj():
    if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
fig.savefig(OUT + "figure2_by_network_z_rho_delta.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
if PREVIEW: fig.savefig(PREVIEW + "/figure2_by_network_z_rho_delta_grey.png", dpi=110, bbox_inches="tight", facecolor="white")
print(D.groupby(["scenario", "N"]).z_U.median().unstack("N").loc[order].round(2).to_string())
