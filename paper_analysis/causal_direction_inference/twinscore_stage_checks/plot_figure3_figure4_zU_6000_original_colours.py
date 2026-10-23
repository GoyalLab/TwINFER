"""figure3_z_rho_delta_6000_pairs.pdf and figure4_z_rho_delta_6000_pairs.pdf in the style of figure2_z_rho_delta_6000_pairs.pdf (plot_v2.ipynb original scheme): one unfilled box per group, no fliers, lw 2, blue if group median > 1.96, red if < -1.96, grey otherwise, black dashed +-1.96 lines.
Figure 3: figure 3 conditions + the two single-state figure 2 conditions (twin_size_scan_figure3_6000.csv / twin_size_scan_figure2_6000.csv).  Figure 4: Fan_out/Feed_forward/Mutual_regulation x pairs 1-2, 1-3, 2-3, replicates 100-199 (twin_size_scan_figure4_reps100_199.csv), N = 6000.
Illustrator-editable recipe, no title."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.sans-serif": ["Arial"], "font.family": "sans-serif", "svg.fonttype": "none", "mathtext.fontset": "cm", "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/"; O = A + "causal_direction_inference/zrho_pdf_network_auprc/"; OUT = A + "final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; PREVIEW = os.environ.get("PREVIEW_DIR"); TH = 1.96
col = lambda v: "blue" if np.median(v) > TH else ("red" if np.median(v) < -TH else "grey")
def make(vals, labels, name, size):
    fig, ax = plt.subplots(figsize=size); ax.set_facecolor("none"); fig.patch.set_alpha(0); n = len(vals)
    for s_ in (TH, -TH): ax.axhline(s_, linestyle="--", color="black", linewidth=1, alpha=0.7, dashes=(12, 4))
    b = ax.boxplot(vals, positions=range(1, n + 1), patch_artist=True, showfliers=False, widths=0.3)
    for i, v in enumerate(vals):
        c = col(v); b["boxes"][i].set_facecolor("none"); b["boxes"][i].set_edgecolor(c); b["boxes"][i].set_linewidth(2); b["medians"][i].set_color(c); b["medians"][i].set_linewidth(2)
        for w in b["whiskers"][2 * i:2 * i + 2] + b["caps"][2 * i:2 * i + 2]: w.set_color(c); w.set_linewidth(2)
    ax.set_xticks(range(1, n + 1)); ax.set_xticklabels(labels); ax.set_ylabel(r"$z_{\hat\rho_\Delta}$"); ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
    fig.savefig(OUT + name + ".pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    if PREVIEW: fig.savefig(f"{PREVIEW}/{name}_orig.png", dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(name, [(round(float(np.median(v)), 2), col(v)) for v in vals])
d2 = pd.read_csv(O + "twin_size_scan_figure2_6000.csv").rename(columns={"scenario": "g"}); d3 = pd.read_csv(O + "twin_size_scan_figure3_6000.csv").rename(columns={"condition": "g"})
D3 = pd.concat([d2[d2.g.isin(["A_B (single state)", "A_to_B (single state)"])], d3], ignore_index=True)
F3 = ["A_B (single state)", "A_to_B (single state)", "A_rep_B", "A_rep_B_B_rep_A", "A_rep_B_B_to_A", "A_to_B_B_to_A"]; T3 = ["A, B\n(fig 2)", "A→B\n(fig 2)", "A $\\dashv$ B", "A $\\dashv$ B,\nB $\\dashv$ A", "A $\\dashv$ B,\nB → A", "A → B,\nB → A"]
make([D3[D3.g == g].z_U.dropna().values for g in F3], T3, "figure3_z_rho_delta_6000_pairs", (11, 6))
d4 = pd.read_csv(O + "twin_size_scan_figure4_reps100_199.csv"); d4 = d4[d4.N == 6000]; MO = ["Fan_out", "Feed_forward", "Mutual_regulation"]; PR = ["1-2", "1-3", "2-3"]; mn = {"Fan_out": "fan-out", "Feed_forward": "feed-forward", "Mutual_regulation": "regulated\nmutual"}; pl = {"1-2": "1→2", "1-3": "1→3", "2-3": "2–3"}
make([d4[(d4.motif == m) & (d4.pair == p)].z_U.dropna().values for m in MO for p in PR], [f"{mn[m]}\n{pl[p]}" for m in MO for p in PR], "figure4_z_rho_delta_6000_pairs", (16, 6))
