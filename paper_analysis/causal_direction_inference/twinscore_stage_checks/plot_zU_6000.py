"""z_rho_hat_Delta (= z_U) alone with 6,000 twin pairs (all clones, 12,000 cells, 20 h): (1) figure 2 scenarios (signed z-score), (2) network sweep pair classes (network_sweep_final sets, pairs pooled; |z_U|, since edge signs are mixed).
Illustrator-editable recipe (Arial + cm math, guide font sizes, no clips, transparent PDF, no title/description on the plot).   Output: new_metrics_designs/figure2_z_rho_delta_6000_pairs.pdf, network_sweep_by_pair_class_z_rho_delta_6000_pairs.pdf"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42; plt.rcParams["font.sans-serif"] = ["Arial"]; plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["svg.fonttype"] = "none"; plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.labelsize"] = 18; plt.rcParams["axes.titlesize"] = 20; plt.rcParams["xtick.labelsize"] = 12; plt.rcParams["ytick.labelsize"] = 12; plt.rcParams["legend.fontsize"] = 12
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; OUT = A + "analysis_data/paper_analysis/final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; PREVIEW = os.environ.get("PREVIEW_DIR")
BLUE, ORANGE = "#2a78d6", "#eb6834"
def style(ax, ylab, labels, n):
    ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(n)); ax.set_xticklabels(labels); ax.set_xlim(-0.6, n - 0.4); ax.set_ylabel(ylab); ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
def save(fig, name):
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
    fig.savefig(OUT + name + ".pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    if PREVIEW: fig.savefig(f"{PREVIEW}/{name}.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
def boxes(ax, groups, colors, w=0.5):
    for i, (v, c) in enumerate(zip(groups, colors)):
        b = ax.boxplot(v, positions=[i], widths=w, patch_artist=True, showfliers=True, medianprops=dict(color="#0b0b0b", lw=1.5), whiskerprops=dict(color=c, lw=1.1), capprops=dict(color=c, lw=1.1), boxprops=dict(color=c, lw=1.1),
                       flierprops=dict(marker="o", ms=2.6, mfc="none", mec=c, mew=0.8)); b["boxes"][0].set_facecolor(c)
# (1) figure 2 scenarios
D = pd.read_csv(O2 + "twin_size_scan_figure2_6000.csv"); order = ["A_B (single state)", "A_to_B (single state)", "A_B (two states)", "A_to_B (two states)", "A_B (drift)", "A_to_B (drift)"]
lab = ["A, B", "A→B", "A, B\n2 states", "A→B\n2 states", "A, B\ndrift", "A→B\ndrift"]; cols = [BLUE if "A_B" in sc else ORANGE for sc in order]
fig, ax = plt.subplots(figsize=(11, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); boxes(ax, [D[D.scenario == sc].z_U.values for sc in order], cols)
for s_ in (1.96, -1.96): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
style(ax, r"$z_{\hat\rho_\Delta}$", lab, len(order)); ax.legend(handles=[Patch(fc=BLUE, ec=BLUE, label="no regulation"), Patch(fc=ORANGE, ec=ORANGE, label="regulation")], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False); save(fig, "figure2_z_rho_delta_6000_pairs")
print("figure 2, median z_U:", D.groupby("scenario").z_U.median().loc[order].round(2).to_dict()); print("  % |z_U| > 1.96:", (D.assign(a=D.z_U.abs() > 1.96).groupby("scenario").a.mean().loc[order] * 100).round(0).to_dict())
# (2) network sweep classes
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "6000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "cls", "fam"]], on=["raw", "sim", "key"]); Z = Z[Z.fam == "sweep"].dropna(subset=["z_U"]).copy(); Z["azU"] = Z.z_U.abs()
ORD = ["direct edge", "indirect", "co-regulated", "unrelated"]; tick = ["direct edge", "indirect", "co-regulated", "unconnected"]
fig, ax = plt.subplots(figsize=(10, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); boxes(ax, [Z[Z.cls == c].azU.values for c in ORD], [ORANGE, BLUE, BLUE, BLUE])
for s_ in (1.96,): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
style(ax, r"$|z_{\hat\rho_\Delta}|$", tick, len(ORD)); save(fig, "network_sweep_by_pair_class_z_rho_delta_6000_pairs")
t = Z.groupby("cls").azU.agg(n="size", median="median", pct_gt_196=lambda v: 100 * (v > 1.96).mean()).reindex(ORD).round(2); print("\nnetwork sweep (6000 pairs), |z_U| by class:"); print(t.to_string())
# ---- three-term score alone, 6,000 twin pairs: |z_rho| - |z_rho_s| + |z_rho_hat_Delta|  (original z-scores)
D["three"] = D.z_rho.abs() - D.z_C.abs() + D.z_U.abs(); Z["three"] = Z.z_rho.abs() - Z.z_C.abs() + Z.z_U.abs(); ylab3 = r"$|z_\rho| - |z_{\rho_s}| + |z_{\hat\rho_\Delta}|$"
fig, ax = plt.subplots(figsize=(11, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); boxes(ax, [D[D.scenario == sc].three.values for sc in order], cols)
style(ax, ylab3, lab, len(order)); ax.legend(handles=[Patch(fc=BLUE, ec=BLUE, label="no regulation"), Patch(fc=ORANGE, ec=ORANGE, label="regulation")], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False); save(fig, "figure2_three_term_6000_pairs")
print("\nfigure 2, median three-term:", D.groupby("scenario").three.median().loc[order].round(2).to_dict())
fig, ax = plt.subplots(figsize=(10, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); boxes(ax, [Z[Z.cls == c].three.values for c in ORD], [ORANGE, BLUE, BLUE, BLUE]); style(ax, ylab3, tick, len(ORD)); save(fig, "network_sweep_by_pair_class_three_term_6000_pairs")
print("network sweep (6000 pairs), three-term by class:"); print(Z.groupby("cls").three.agg(n="size", median="median").reindex(ORD).round(2).to_string())
