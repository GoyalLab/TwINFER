"""Network sweep (network_sweep_final sets, all pairs pooled, 3,000 twin pairs at 20 h): box plots of the two-term score |z_rho|-|z_C|, the three-term score |z_rho|-|z_C|+|z_U| and |z_U| (= |z_rho_hat_Delta|) for the four pair classes
(direct edge, indirect, co-regulated, unconnected).  Original z-scores.  Follows the Illustrator-editable recipe (Arial + cm math, guide font sizes, no clips, transparent PDF, no title/description on the plot).
Usage: python plot_network_classes.py   (PREVIEW_DIR env var writes a PNG preview elsewhere)"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42; plt.rcParams["font.sans-serif"] = ["Arial"]; plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["svg.fonttype"] = "none"; plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.labelsize"] = 18; plt.rcParams["axes.titlesize"] = 20; plt.rcParams["xtick.labelsize"] = 12; plt.rcParams["ytick.labelsize"] = 12; plt.rcParams["legend.fontsize"] = 12; plt.rcParams["legend.title_fontsize"] = 12
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])      # loader: pair tables with class labels (d: raw, sim, x, y, cls, fam)
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; OUT = A + "analysis_data/paper_analysis/final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; PREVIEW = os.environ.get("PREVIEW_DIR")
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design.isin(["3000 twins", "6000 twins"])].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "cls", "fam"]], on=["raw", "sim", "key"]); Z = Z[Z.fam == "sweep"].dropna(subset=["z_rho", "z_C", "z_U"]).copy()
Z["N"] = Z.design.str.extract(r"(\d+)")[0].astype(int); Z["two"] = Z.z_rho.abs() - Z.z_C.abs(); Z["three"] = Z.two + Z.z_U.abs(); Z["absU"] = Z.z_U.abs()
ORDER = ["direct edge", "indirect", "co-regulated", "unrelated"]; TICK = ["direct edge", "indirect", "co-regulated", "unconnected"]
SC = [("two", "#2a78d6", r"$|z_\rho| - |z_{\rho_s}|$"), ("three", "#eb6834", r"$|z_\rho| - |z_{\rho_s}| + |z_{\hat\rho_\Delta}|$"), ("absU", "#1baf7a", r"$|z_{\hat\rho_\Delta}|$")]
def make(N):
    ZN = Z[Z.N == N]
    fig, ax = plt.subplots(figsize=(13, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); w = 0.22
    for i, cl in enumerate(ORDER):
        for j, (col, c, lab) in enumerate(SC):
            v = ZN[ZN.cls == cl][col].dropna().values
            b = ax.boxplot(v, positions=[i + (j - 1) * w * 1.15], widths=w, patch_artist=True, showfliers=True, medianprops=dict(color="#0b0b0b", lw=1.4), whiskerprops=dict(color=c, lw=1.1), capprops=dict(color=c, lw=1.1),
                           boxprops=dict(color=c, lw=1.1), flierprops=dict(marker="o", ms=2.2, mfc="none", mec=c, mew=0.7)); b["boxes"][0].set_facecolor(c)
    ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(TICK); ax.set_xlim(-0.55, len(ORDER) - 0.45); ax.set_ylabel("score"); ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
    ax.legend(handles=[Patch(fc=c, ec=c, label=lab) for _, c, lab in SC], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
    fig.savefig(OUT + f"network_sweep_by_pair_class_score_comparison_{N}_pairs.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    if PREVIEW: fig.savefig(PREVIEW + f"/network_sweep_by_pair_class_score_comparison_{N}_pairs.png", dpi=110, bbox_inches="tight", facecolor="white")
    t = ZN.groupby("cls")[["two", "three", "absU"]].agg(["median"]).reindex(ORDER).round(2); t.columns = ["two-term", "three-term", "|z_U|"]; t.insert(0, "n_pairs", ZN.groupby("cls").size().reindex(ORDER)); print(f"N = {N} twin pairs"); print(t.to_string())
    plt.close(fig)

for N in (3000, 6000): make(N)
