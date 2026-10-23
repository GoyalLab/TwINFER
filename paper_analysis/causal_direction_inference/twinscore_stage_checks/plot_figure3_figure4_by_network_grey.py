"""Greyscale twin-count box plots grouped by network/condition for figure 3 and figure 4 (same style as figure2_by_network_*.pdf): z_rho_delta (z_U), z_rho_s (z_C) and z_rho, 250/500/1000/3000/6000 twin pairs (light -> dark).
Figure 3: conditions A rep. B, A rep. B & B rep. A, A rep. B & B->A, A->B & B->A (+ the two single-state figure 2 conditions).  Figure 4: Fan_out, Feed_forward, Mutual_regulation x pairs 1-2, 1-3, 2-3.
Needs twin_size_scan_fig3_fig4.py outputs.  Illustrator-editable recipe; no titles.  Output: new_metrics_designs/figure{3,4}_by_network_{z_rho_delta,z_rho_s,z_rho}.pdf"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.sans-serif": ["Arial"], "font.family": "sans-serif", "svg.fonttype": "none", "mathtext.fontset": "cm", "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "legend.title_fontsize": 12})
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/"; O = A + "causal_direction_inference/zrho_pdf_network_auprc/"; OUT = A + "final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; PREVIEW = os.environ.get("PREVIEW_DIR")
NS = [250, 500, 1000, 3000, 6000]; GREY = ["#b8b8b8", "#9a9a9a", "#7c7c7c", "#4f4f4f", "#1a1a1a"]; MEDIAN = ["#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff", "#ffffff"]
STATS = [("z_U", r"$z_{\hat\rho_\Delta}$", "z_rho_delta"), ("z_C", r"$z_{\rho_s}$", "z_rho_s"), ("z_rho", r"$z_\rho$", "z_rho")]
def figure(D, key, order, labels, prefix, size, w):
    for col, ylab, fname in STATS:
        fig, ax = plt.subplots(figsize=size); ax.set_facecolor("none"); fig.patch.set_alpha(0); k = len(NS)
        for i, sc in enumerate(order):
            for j, N in enumerate(NS):
                v = D[(D[key] == sc) & (D.N == N)][col].dropna().values
                b = ax.boxplot(v, positions=[i + (j - (k - 1) / 2) * w], widths=w * 0.88, patch_artist=True, showfliers=True, medianprops=dict(color=MEDIAN[j], lw=1.4), whiskerprops=dict(color="#0b0b0b", lw=0.9), capprops=dict(color="#0b0b0b", lw=0.9),
                               boxprops=dict(color="#0b0b0b", lw=0.8), flierprops=dict(marker="o", ms=2.4, mfc="none", mec="#52514e", mew=0.7)); b["boxes"][0].set_facecolor(GREY[j])
        if col != "z_rho" or True:
            for s_ in (1.96, -1.96): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(len(order))); ax.set_xticklabels(labels); ax.set_xlim(-0.55, len(order) - 0.45); ax.set_ylabel(ylab); ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
        ax.legend(handles=[Patch(fc=GREY[j], ec="#0b0b0b", lw=0.8, label=str(NS[j])) for j in range(k)], title="number of twin pairs", loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, columnspacing=1.4, handlelength=1.4)
        for artist in fig.findobj():
            if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
        fig.savefig(f"{OUT}{prefix}_by_network_{fname}.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
        if PREVIEW: fig.savefig(f"{PREVIEW}/{prefix}_by_network_{fname}_grey.png", dpi=90, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(prefix, fname, D.groupby([key, "N"])[col].median().unstack("N").loc[order].round(2).to_string(), sep="\n")
if __name__ == "__main__":
    d2 = pd.concat([pd.read_csv(O + "twin_size_scan_figure2.csv").query("N in @NS"), pd.read_csv(O + "twin_size_scan_figure2_6000.csv")], ignore_index=True)
    d3 = pd.concat([pd.read_csv(O + "twin_size_scan_figure3.csv").query("N in @NS").rename(columns={"condition": "scenario"}), pd.read_csv(O + "twin_size_scan_figure3_6000.csv").rename(columns={"condition": "scenario"}), d2[d2.scenario.isin(["A_B (single state)", "A_to_B (single state)"])]], ignore_index=True)
    F3 = ["A_B (single state)", "A_to_B (single state)", "A_rep_B", "A_rep_B_B_rep_A", "A_rep_B_B_to_A", "A_to_B_B_to_A"]; T3 = ["A, B\n(fig 2)", "A→B\n(fig 2)", r"A $\dashv$ B", "A $\\dashv$ B,\nB $\\dashv$ A", "A $\\dashv$ B,\nB → A", "A → B,\nB → A"]      # repression drawn as -| (cm math)
    figure(d3, "scenario", F3, T3, "figure3", (13, 6), 0.16)
    d4 = pd.read_csv(O + os.environ.get("FIG4_CSV", "twin_size_scan_figure4_reps100_199.csv")); d4["g"] = d4.motif + " " + d4.pair
    pl = {"1-2": "1→2", "1-3": "1→3", "2-3": "2–3"}; mn = {"Fan_out": "fan-out", "Feed_forward": "feed-forward", "Mutual_regulation": "regulated\nmutual"}
    O4 = [f"{m} {p}" for m in ["Fan_out", "Feed_forward", "Mutual_regulation"] for p in ["1-2", "1-3", "2-3"]]; T4 = [f"{mn[m]}\n{pl[p]}" for m in ["Fan_out", "Feed_forward", "Mutual_regulation"] for p in ["1-2", "1-3", "2-3"]]
    figure(d4, "g", O4, T4, "figure4", (22, 6), 0.16)
