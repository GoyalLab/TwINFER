"""Box plots of the TwinScore z-scores, in the style of paper_analysis/heterogeneity_vs_regulation/plot_v2.ipynb
(unfilled coloured boxes, hatching for the second group, transparent background, Illustrator-friendly fonts, no clipping, dashed black thresholds; pdf + svg + png).
Two separate sets of figures, one figure per z-score in each set:
  2gene: A, B (no edge) and A->B (edge); single state, 2 states, drift  (results/rep_*.csv, results_drift/*_rep_*.csv; 100 replicates each)
  3gene: Fan out, Feed forward, Regulated mutual; pairs 1-2, 1-3, 2-3 of each motif  (results_motif_design1500/<motif>_rep_*.csv; 50 replicates each)
z-scores: z_rho (Stage I), z_rho_s (shared part = C, the h-stat), z_rho_hat_Delta, z_dagger(x->y), z_dagger(y->x).
Design everywhere: sample B = 20 h (1,500 twin clones + 3,000 singletons), sample A = 1 h, 3,000 shared clones, mRNA only.
Outputs go to <data root>/paper_analysis/final_plots/plot_z_Scores_2_gene_3_gene/ (never into the repo).
Replaced in the 2026-10-08 rewrite (v1 kept in the session scratchpad): one combined plot of the 2-gene types and Fan_out; labels were A_B, A_to_B, A_B_2_states, Fan_out ..."""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from twinfer.utils.paths import get_data_root

ap = argparse.ArgumentParser()
ap.add_argument("--no-drift", action="store_true", help="leave out the drift types in the 2-gene plots")
ap.add_argument("--linear", action="store_true", help="linear y axis (default: symmetric log, linear within +-4)")
args = ap.parse_args()

DATA = get_data_root()
IN = DATA / "paper_analysis" / "causal_direction_inference" / "figure2_1000_zscores"
OUT = DATA / "paper_analysis" / "final_plots" / "plot_z_Scores_2_gene_3_gene"; OUT.mkdir(parents=True, exist_ok=True)

# ---- fonts / style (as in plot_v2.ipynb) ----
try:
    import twinfer.plotting as twinfer_plotting
    for fp in sorted((Path(twinfer_plotting.__file__).parent / "assets" / "fonts").glob("*.ttf")): fm.fontManager.addfont(fp)
except Exception as e: print("font assets not loaded:", e)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "mathtext.fontset": "cm", "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 13, "ytick.labelsize": 14, "legend.fontsize": 12})

# ---- data ----
two = pd.concat([pd.read_csv(f) for f in glob.glob(str(IN / "results" / "rep_*.csv")) + glob.glob(str(IN / "results_drift" / "*_rep_*.csv"))])
motifs = pd.concat([pd.read_csv(f) for f in glob.glob(str(IN / "results_motif_design1500" / "*_rep_*.csv"))])

# ---- labels ----
AB = "A, B"; ATOB = r"$\mathrm{A\rightarrow B}$"
GENE2 = [("A_B (single state)", AB, "#bc2462", False), ("A_to_B (single state)", ATOB, "#754c29", False),
         ("A_B (two states)", AB + "\n2 states", "#ee8d39", True), ("A_to_B (two states)", ATOB + "\n2 states", "#006838", True)]
if not args.no_drift: GENE2 += [("A_B (drift)", AB + "\ndrift", "#2f6fa3", True), ("A_to_B (drift)", ATOB + "\ndrift", "#7b4ea3", True)]
# 3 genes: (motif key, display name, [(pair, tick label, colour, hatched)])
EDGE, NOEDGE = "#754c29", "#bc2462"
GENE3 = [("Fan_out", "Fan out", [("1-2", r"$1\rightarrow2$", EDGE, False), ("1-3", r"$1\rightarrow3$", EDGE, False), ("2-3", r"$2$, $3$", NOEDGE, True)]),
         ("Feed_forward", "Feed forward", [("1-2", r"$1\rightarrow2$", EDGE, False), ("1-3", r"$1\rightarrow3$", EDGE, False), ("2-3", r"$2\rightarrow3$", EDGE, True)]),
         ("Mutual_regulation", "Regulated mutual", [("1-2", r"$1\rightarrow2$", EDGE, False), ("1-3", r"$1\rightarrow3$", EDGE, False), ("2-3", r"$2\leftrightarrow3$", EDGE, True)])]
PANELS = [("z_rho", r"$z_{\rho}$ (Stage I)", 1.96, "z_rho"), ("z_C", r"$z_{\rho_s}$ (shared part, h-stat)", 1.96, "z_C"),
          ("z_U", r"$z_{\hat{\rho}_{\Delta}}$", 1.96, "z_U"), ("z_dag_xy", r"$z^{\dagger}(x \rightarrow y)$", 2.576, "z_dag_xy"), ("z_dag_yx", r"$z^{\dagger}(y \rightarrow x)$", 2.576, "z_dag_yx")]

def groups_2gene(col):
    return [(lab, c, h, two.loc[two.scenario == scen, col].dropna().values) for scen, lab, c, h in GENE2], list(range(len(GENE2))), []
def groups_3gene(col):
    G, pos, blocks, x = [], [], [], 0
    for key, name, pairs in GENE3:
        first = x
        for pair, lab, c, h in pairs:
            G.append((lab, c, h, motifs.loc[(motifs.motif == key) & (motifs.pair == pair), col].dropna().values)); pos.append(x); x += 1
        blocks.append((name, (first + x - 1) / 2, first - 0.5)); x += 1
    return G, pos, blocks

def draw(ax, G, pos, blocks, ylabel, thr):
    box = ax.boxplot([g[3] for g in G], positions=pos, widths=0.6, patch_artist=True, showfliers=True)
    for i, (lab, c, hatched, v) in enumerate(G):
        b = box["boxes"][i]; b.set_facecolor("none"); b.set_edgecolor(c); b.set_linewidth(1.0)
        if hatched: b.set_hatch("///")
        wc = box["whiskers"][2 * i:2 * i + 2] + box["caps"][2 * i:2 * i + 2]
        for w in wc: w.set_color(c); w.set_linewidth(1.0); w.set_clip_on(False)
        box["medians"][i].set_color(c); box["medians"][i].set_linewidth(1.5)
        fl = box["fliers"][i]; fl.set_markerfacecolor("none"); fl.set_markeredgecolor(c); fl.set_markersize(4)
        for el in (b, box["medians"][i], fl): el.set_clip_on(False)
    for s in (thr, -thr): ax.axhline(s, linestyle="--", color="black", linewidth=1, alpha=0.7)
    ax.axhline(0, linestyle=":", color="black", linewidth=0.8, alpha=0.5)
    if not args.linear:
        ax.set_yscale("symlog", linthresh=4, linscale=1.2); ax.set_yticks([-30, -10, -4, -2, 0, 2, 4, 10, 30]); ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(pos); ax.set_xticklabels([g[0] for g in G]); ax.set_ylabel(ylabel)
    for name, xc, xsep in blocks:
        ax.text(xc, -0.13, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=15)
        if xsep > 0: ax.axvline(xsep - 0.5, color="grey", linewidth=0.6, alpha=0.5)
    ax.set_facecolor("none"); ax.grid(False)
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def draw_dual(ax, getter, thr, ylabel):
    """z_dagger(x->y) and z_dagger(y->x) side by side in every category: x->y = open box, y->x = same colour, light fill."""
    G1, pos, blocks = getter("z_dag_xy"); G2, _, _ = getter("z_dag_yx"); off, w = 0.21, 0.36
    for G, d, shaded in ((G1, -off, False), (G2, +off, True)):
        box = ax.boxplot([g[3] for g in G], positions=[x + d for x in pos], widths=w, patch_artist=True, showfliers=True)
        for i, (lab, c, hatched, v) in enumerate(G):
            b = box["boxes"][i]; b.set_facecolor(matplotlib.colors.to_rgba(c, 0.28) if shaded else "none"); b.set_edgecolor(c); b.set_linewidth(1.0)
            if hatched: b.set_hatch("///")
            wc = box["whiskers"][2 * i:2 * i + 2] + box["caps"][2 * i:2 * i + 2]
            for e in wc: e.set_color(c); e.set_linewidth(1.0); e.set_clip_on(False)
            box["medians"][i].set_color(c); box["medians"][i].set_linewidth(1.5)
            fl = box["fliers"][i]; fl.set_markerfacecolor("none"); fl.set_markeredgecolor(c); fl.set_markersize(4)
            for el in (b, box["medians"][i], fl): el.set_clip_on(False)
    for s_ in (thr, -thr): ax.axhline(s_, linestyle="--", color="black", linewidth=1, alpha=0.7)
    ax.axhline(0, linestyle=":", color="black", linewidth=0.8, alpha=0.5)
    if not args.linear:
        ax.set_yscale("symlog", linthresh=4, linscale=1.2); ax.set_yticks([-30, -10, -4, -2, 0, 2, 4, 10, 30]); ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(pos); ax.set_xticklabels([g[0] for g in G1]); ax.set_ylabel(ylabel)
    for name, xc, xsep in blocks:
        ax.text(xc, -0.13, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=15)
        if xsep > 0: ax.axvline(xsep - 0.5, color="grey", linewidth=0.6, alpha=0.5)
    ax.set_facecolor("none"); ax.grid(False)
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=4, width=1)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

def legend_for(kind):
    th = Line2D([0], [0], color="black", linestyle="--", label=r"single-pair threshold (1.96; 2.576 for $z^{\dagger}$)")
    if kind == "2gene":
        return [Patch(facecolor="none", edgecolor="black", label="single state"), Patch(facecolor="none", edgecolor="black", hatch="///", label="2 states or drift"), th]
    return [Patch(facecolor="none", edgecolor=EDGE, label="direct edge"), Patch(facecolor="none", edgecolor=NOEDGE, hatch="///", label="no direct edge (shared driver gene 1)"),
            Patch(facecolor="none", edgecolor=EDGE, hatch="///", label="direct edge, plus shared driver gene 1"), th]

def foot(kind):
    if kind == "2gene":
        n = int(two.groupby("scenario").size().min())
        return f"{n} replicates per type; B = 20 h (1,500 twin clones + 3,000 singletons), A = 1 h, 3,000 shared clones; mRNA"
    n = int(motifs.groupby(["motif", "pair"]).size().min())
    return f"{n} replicates per motif; genes 1, 2, 3; pair 2-3 has gene 1 as shared driver; B = 20 h (1,500 twin clones + 3,000 singletons), A = 1 h, 3,000 shared clones; mRNA"

for kind, getter, size in (("2gene", groups_2gene, (9, 5.5)), ("3gene", groups_3gene, (11, 5.5))):
    for col, ylabel, thr, name in PANELS:
        G, pos, blocks = getter(col); fig, ax = plt.subplots(figsize=size); fig.patch.set_facecolor("none"); draw(ax, G, pos, blocks, ylabel, thr)
        ax.legend(handles=legend_for(kind), loc="lower left", frameon=False, fontsize=10)
        fig.text(0.01, 0.005, foot(kind), fontsize=8, color="grey"); plt.tight_layout(rect=(0, 0.02, 1, 1))
        for ext in ("png", "pdf", "svg"): fig.savefig(OUT / f"zscore_boxplot_{kind}_{name}.{ext}", format=ext, bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True, dpi=200)
        plt.close(fig)
    # both directions in one plot (x->y open, y->x shaded) for easy comparison
    fig, ax = plt.subplots(figsize=(size[0] + 2, size[1])); fig.patch.set_facecolor("none")
    draw_dual(ax, getter, 2.576, r"$z^{\dagger}$ (direction)")
    hd = [Patch(facecolor="none", edgecolor="black", label=r"$z^{\dagger}(x\rightarrow y)$: x in A (1 h), y in B (20 h)"), Patch(facecolor=matplotlib.colors.to_rgba("black", 0.28), edgecolor="black", label=r"$z^{\dagger}(y\rightarrow x)$: y in A (1 h), x in B (20 h)"),
          Line2D([0], [0], color="black", linestyle="--", label=r"threshold 2.576")] + legend_for(kind)[:-1]
    ax.legend(handles=hd, loc="lower left", frameon=False, fontsize=9, ncol=2)
    fig.text(0.01, 0.005, foot(kind), fontsize=8, color="grey"); plt.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf", "svg"): fig.savefig(OUT / f"zscore_boxplot_{kind}_z_dag_both.{ext}", format=ext, bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True, dpi=200)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(size[0] * 3, size[1] * 2 + 1)); fig.patch.set_facecolor("white"); axs = axs.ravel()   # overview of all five z-scores
    for ax, (col, ylabel, thr, name) in zip(axs, PANELS):
        G, pos, blocks = getter(col); draw(ax, G, pos, blocks, ylabel, thr)
    axs[5].axis("off"); axs[5].legend(handles=legend_for(kind), loc="upper left", frameon=False, fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT / f"zscore_boxplots_{kind}_overview.png", dpi=130, facecolor="white")
    fig.savefig(OUT / f"zscore_boxplots_{kind}_all_zscores.pdf", format="pdf", facecolor="white")   # one single-page PDF per set: z_rho, z_rho_s, z_rho_hat_Delta, z_dag(x->y), z_dag(y->x) + legend
    plt.close(fig)
print("saved to", OUT)

# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)] # ---- ONE page with all z-scores: 2-gene set (top two rows) and 3-gene set (bottom two rows), each in the 2x3 arrangement (5 z-scores + legend) ----
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)] fig, axs = plt.subplots(4, 3, figsize=(30, 28)); fig.patch.set_facecolor("white")
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)] for r0, (kind, getter, title) in zip((0, 2), (("2gene", groups_2gene, "2 genes"), ("3gene", groups_3gene, "3 genes: Fan out, Feed forward, Regulated mutual"))):
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)]     block = axs[r0:r0 + 2].ravel()
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)]     for ax, (col, ylabel, thr, name) in zip(block, PANELS):
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)]         G, pos, blocks = getter(col); draw(ax, G, pos, blocks, ylabel, thr)
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)]     block[5].axis("off"); block[5].legend(handles=legend_for(kind), loc="upper left", frameon=False, fontsize=16)
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)]     block[5].text(0.0, 0.35, foot(kind), transform=block[5].transAxes, fontsize=11, color="grey", wrap=True, va="top")
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)]     axs[r0, 0].annotate(title, xy=(0, 1.12), xycoords="axes fraction", fontsize=26, fontweight="bold", va="bottom")
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)] plt.tight_layout(h_pad=3.0, rect=(0, 0, 1, 0.985))
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)] for ext in ("pdf", "png"): fig.savefig(OUT / f"zscore_boxplots_all_one_page.{ext}", format=ext, bbox_inches="tight", facecolor="white", dpi=110)
# [2026-10-08 commented out: replaced by one single-page PDF per set (2 genes / 3 genes separately)] plt.close(fig); print("one-page overview saved")
