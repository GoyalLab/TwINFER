#!/usr/bin/env python3
"""Dot plot of annotate_bins_and_tf_targets.py results for FM01 (left) and FM06 (right): rows = the Venn bins (top block) and TFs (bottom block), columns = MSigDB Hallmark sets (those with q < 0.05 in >= 2 lists of either dataset, max N_COLS) plus the Reactome pathway
'MITF-M-dependent Gene Expression'.  Dot shown only for q < 0.05; size = genes of the list in the set, colour = -log10 q.  Input: annotation_<ds>_{bins,tfs}.csv.   Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/annotation_dotplot_FM01_FM06.{pdf,png}   usage: annotation_dotplot.py [DATASETS... [--out NAME] [--prefix annotation_mouse]]  (--prefix annotation_mouse reads the mouse-native tables of annotate_bins_and_tf_targets_mouse.py)  (default FM01 FM06; for other datasets the columns are the Hallmark/Reactome sets most often significant, N_COLS of them)"""
import sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; O = f"{A}/fm_meta_programs_comparison"; N_COLS = 14; MITFM = "MITF-M-dependent Gene Expression"
args = sys.argv[1:]; OUTN = args[args.index("--out") + 1] if "--out" in args else None; PFX = args[args.index("--prefix") + 1] if "--prefix" in args else "annotation"; rest = [a for i, a in enumerate(args) if a not in ("--out", "--prefix") and not (i > 0 and args[i - 1] in ("--out", "--prefix"))]; DSL = rest or ["FM01", "FM06"]; MEL = DSL == ["FM01", "FM06"]
R = {ds: pd.concat([pd.read_csv(f"{O}/{PFX}_{ds}_{l}.csv") for l in ("bins", "tfs")]) for ds in DSL}
R = {ds: d[(d.q < 0.05) & (((d.library == "Hallmark")) | ((d.library == "Reactome") & ((d.term == MITFM) if MEL else True)))] for ds, d in R.items()}
cnt = pd.concat(R.values()).groupby("term").name.count().sort_values(ascending=False); FORCE = ["E2F Targets", "G2-M Checkpoint", "Mitotic Spindle", "Myc Targets V1", "Myc Targets V2"] if MEL else []; hall = FORCE + [t for t in cnt.index if t != MITFM and t not in FORCE][:N_COLS - len(FORCE)]; terms = hall + ([MITFM] if MEL else [])
# column order: cell-cycle-like first, then the rest by overall count
cc = [t for t in terms if t in FORCE]; terms = cc + [t for t in terms if t not in cc]
SH = {"Epithelial Mesenchymal Transition": "EMT", "TNF-alpha Signaling via NF-kB": "TNF-alpha / NF-kB", MITFM: "MITF-M targets (Reactome)", "Epithelial Mesenchymal Transition": "EMT"}
def rows(ds):
    d = pd.read_csv(f"{O}/annotation_{ds}_bins.csv"); sm = pd.concat([pd.read_csv(p) for p in glob.glob(f"{O}/{PFX}_summary*.csv")]).drop_duplicates(["dataset", "level", "name"]); sm = sm[sm.dataset == ds]
    b = [n for n in sm[sm.level == "bins"].name]; t = sorted(sm[sm.level == "tfs"].name); pretty = lambda n: n.replace("rho_D", "ρΔ").replace("rho_s", "ρs").replace("rho", "ρ")
    t = [x for x in t if ((R[ds].name == x) & R[ds].term.isin(terms)).any()]
    return b, t, pretty
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf"); plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
YM = []
fig, axs = plt.subplots(1, len(DSL), figsize=(8.75 * len(DSL), 7.4 if MEL else 8.4), squeeze=False, gridspec_kw={"wspace": 0.5}); axs = axs[0]; fig.patch.set_alpha(0); sc = None
for ax, ds in zip(axs, DSL):
    b, t, pretty = rows(ds); labs = [pretty(x) for x in b] + [None] + t; ypos = {}; y = 0
    for x in b: ypos[x] = y; y += 1
    y += 0.7
    for x in t: ypos[x] = y; y += 1
    xs, ys, ss, cc_ = [], [], [], []
    for r in R[ds].itertuples():
        if r.term in terms and r.name in ypos: xs.append(terms.index(r.term)); ys.append(ypos[r.name]); ss.append(18 + 17 * r.k); cc_.append(-np.log10(max(r.q, 1e-12)))
    ax.set_facecolor("none"); sc = ax.scatter(xs, ys, s=ss, c=cc_, cmap="Purples", vmin=1.3, vmax=12, edgecolor="#555", linewidth=0.4, zorder=3)
    ax.axhline(len(b) - 0.5 + 0.35, color="#ccc", lw=0.8); ax.set_yticks([ypos[x] for x in b + t]); ax.set_yticklabels([pretty(x) for x in b] + t, fontsize=9.3); ax.set_xticks(range(len(terms))); ax.set_xticklabels([(lambda x: x if len(x) <= 34 else x[:32] + "...")(SH.get(t, t)) for t in terms], rotation=50, ha="right", fontsize=9)
    YM.append(y); ax.set_ylim(y - 0.4, -0.7); ax.set_xlim(-0.6, len(terms) - 0.4); ax.set_title(ds, fontsize=12, fontweight="bold", loc="left"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="x", color="#f0f0f0", lw=0.8, zorder=0)
[a.set_ylim(max(YM) - 0.4, -0.7) for a in axs]
cax = fig.add_axes([0.945, 0.55, 0.008, 0.25]); cb = fig.colorbar(sc, cax=cax); cb.set_label("-log10 q", fontsize=9); cb.ax.tick_params(labelsize=8)
hs = [plt.scatter([], [], s=18 + 17 * n, c="#bbb", edgecolor="#555", linewidth=0.4) for n in (2, 5, 15)]
fig.legend(hs, ["2 genes", "5 genes", "15 genes"], loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.45, -0.06))
fig.text(0.005, 0.99, "Top block of each panel: Venn bins (target genes of each region); bottom block: TFs with >= 15 targets in the top-1000 shared/unshared pairs. " + ("MSigDB Hallmark 2020 (+ Reactome MITF-M), custom background = dataset targets, q < 0.05; TFs without any significant dot not shown." if MEL and PFX == "annotation" else ("MOUSE-native sets: MSigDB mouse Hallmark + mouse Reactome; background = the dataset's mapped targets annotated in the library; q < 0.05; TFs without any significant dot not shown." if PFX != "annotation" else "MSigDB Hallmark 2020 + Reactome (human), custom background = dataset targets, q < 0.05; TFs without any significant dot not shown.")), fontsize=8, color="#777", ha="left", va="top")
fig.subplots_adjust(left=0.07, right=0.925, bottom=0.27, top=0.92)
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{O}/{PFX.replace('annotation', 'annotation_dotplot')}_" + (OUTN or "_".join(DSL)); fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight", pad_inches=0.08); fig.savefig(out + ".png", dpi=105, facecolor="white", bbox_inches="tight", pad_inches=0.08); print("saved", out, terms)
