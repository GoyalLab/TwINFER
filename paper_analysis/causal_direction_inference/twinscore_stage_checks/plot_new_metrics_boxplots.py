"""Box plots over 100 replicates x 6 scenarios (A,B | A->B  x  single state | 2 states | drift) of z_rho, z_C, z_U and the new combined score, for two sampling designs of the 20 h sample:
  X  = 1,500 twin clones + 3,000 singletons (existing per-replicate tables, scenario_table_pictures_figure2.csv)
  Y3 = 3,000 twin clones, no singletons (computed here from the same simulation files; same two-state mixture as X; random 3,000 of the 6,000 clones)
Second figure (--random-pairs): the same two designs but the "twins" are RANDOM cell pairs (sister pairing broken: every clone's second cell is swapped with the second cell of another clone; cells and marginals identical).
No cross-sample terms (no z_dagger, no 1 h sample).  New score: |z_rho| - |z_C| + |z_U|.
Output: analysis_data/paper_analysis/final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/  (pdf + png);  cache: <O>/new_metrics_3000twins_scenarios.csv
usage: python plot_new_metrics_boxplots.py [--jobs 8]"""
import argparse, glob, os, numpy as np, pandas as pd
from figure2_1000_zscores import NCL, G, SCEN, find, analyse
from u_time_pool import zU
from motif_replicates_zscores import zC, zRho
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/"; O = A + "causal_direction_inference/zrho_pdf_network_auprc/"; Z = A + "causal_direction_inference/figure2_1000_zscores/"
OUTDIR = A + "final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; CACHE = O + "new_metrics_3000twins_scenarios.csv"; CACHE_R = O + "new_metrics_randompairs_scenarios.csv"
DRIFT = "/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/"
ORDER = ["A_B (single state)", "A_to_B (single state)", "A_B (two states)", "A_to_B (two states)", "A_B (drift)", "A_to_B (drift)"]
def load20(path):
    d = pd.read_csv(path, usecols=["time_step", "gene_1_mRNA", "gene_2_mRNA", "replicate", "clone_id"]); s = d[d.time_step == 20]
    return {g: s.pivot(index="clone_id", columns="replicate", values=f"{g}_mRNA").values for g in G}
def run_rep(rep):
    rows = []; jobs = [(name, cfgs, None, si) for si, (name, cfgs) in enumerate(SCEN)] + [(f"{n} (drift)", None, n, 7) for n in ("A_B", "A_to_B")]
    for name, cfgs, net, si in jobs:
        rng = np.random.RandomState(rep * 10 + si)
        if net: P = load20(glob.glob(f"{DRIFT}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")[0])
        elif len(cfgs) == 1: P = load20(find(cfgs[0], rep)[0])
        else:
            S = [load20(find(c, rep)[0]) for c in cfgs]; sa, sb = rng.choice(NCL, 3000, replace=False), rng.choice(NCL, 3000, replace=False)   # same draws as figure2_1000_zscores.run_rep
            P = {g: np.vstack([S[0][g][sa], S[1][g][sb]]) for g in G}
        c = np.random.RandomState(rep * 100 + si + 12345).permutation(NCL)[:3000]; X, Y = P["gene_1"][c], P["gene_2"][c]
        S_, C_, zc = zC(X, Y); rows.append(dict(scenario=name, rep=rep, design="3000 twins", z_rho=zRho(X, Y), z_C=zc, z_U=zU(X, Y)))
    return rows
def run_rep_random(rep):
    rows = []; jobs = [(name, cfgs, None, si) for si, (name, cfgs) in enumerate(SCEN)] + [(f"{n} (drift)", None, n, 7) for n in ("A_B", "A_to_B")]
    for name, cfgs, net, si in jobs:
        rng = np.random.RandomState(rep * 10 + si)
        if net: P = load20(glob.glob(f"{DRIFT}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")[0])
        elif len(cfgs) == 1: P = load20(find(cfgs[0], rep)[0])
        else:
            S = [load20(find(c, rep)[0]) for c in cfgs]; sa, sb = rng.choice(NCL, 3000, replace=False), rng.choice(NCL, 3000, replace=False)
            P = {g: np.vstack([S[0][g][sa], S[1][g][sb]]) for g in G}
        perm = np.random.RandomState(rep * 100 + si + 777).permutation(NCL); pi = np.empty(NCL, int); pi[perm] = np.roll(perm, -1)          # cyclic shift of a random permutation = derangement
        Pr = {g: np.column_stack([P[g][:, 0], P[g][pi, 1]]) for g in G}                                                                 # same pi for both genes: cells stay intact
        o = analyse(Pr, Pr, np.random.RandomState(rep * 10 + si + 999), 2)                                                               # design X on random pairs (cross-sample outputs ignored)
        rows.append(dict(scenario=name, rep=rep, design="1500 random pairs + 3000 singletons", z_rho=o["z_rho"], z_C=o["z_C"], z_U=o["z_U"]))
        c = np.random.RandomState(rep * 100 + si + 12345).permutation(NCL)[:3000]; X, Y = Pr["gene_1"][c], Pr["gene_2"][c]; S_, C_, zc = zC(X, Y)
        rows.append(dict(scenario=name, rep=rep, design="3000 random pairs", z_rho=zRho(X, Y), z_C=zc, z_U=zU(X, Y)))
    return rows
def make_figure(D, designs, title, outname, legend_names, note_extra="", standardized=False, order=None, labels=None, colors=None, panels=None, scen_col="scenario"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import font_manager as fm; from matplotlib.patches import Patch; from matplotlib.colors import to_rgba, to_rgb
    plt.rcParams["hatch.linewidth"] = 0.9
    for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.sans-serif": ["Arial"], "font.family": "sans-serif", "mathtext.fontset": "cm",
                         "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})   # [2026-10-17 user's Illustrator-editable recipe, memory feedback_illustrator_editable_plots]
    D = D.copy()
    if "new3" not in D: D["new3"] = D.z_rho.abs() - D.z_C.abs() + D.z_U.abs()
    INK, SUB, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    # opaque pre-blended fill (colour at `alpha` over the surface): in PDF, a hatch tile painted with the same colour as its lines, or with alpha, hides the hatching
    blend = lambda col, al: tuple(al * x + (1 - al) * y for x, y in zip(to_rgb(col), to_rgb(SURF)))
    order = order or ORDER; COL = dict(zip(order, colors or ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))      # categorical slots 1-6 (validated light palette)
    LAB = labels or ["A, B", "A→B", "A, B\n2 states", "A→B\n2 states", "A, B\ndrift", "A→B\ndrift"]
    DESIGNS = [(designs[0], -0.2, 0.30, False), (designs[1], 0.2, 0.30, True)]     # [2026-10-17 design = hatch (second design hatched); 2-states/drift no longer hatched]
    PANELS = [("z_rho", r"$z_{\rho}$ (Stage I)", 1.96, (-45, 45)), ("z_C", r"$z_{\rho_s}$ (shared part, h-stat)", 1.96, (-45, 45)), ("z_U", r"$z_{\hat\rho_\Delta}$ (unshared)", 1.96, (-45, 45)),
              ("new3", r"$|z_\rho| - |z_{\rho_s}| + |z_{\hat\rho_\Delta}|$", None, (-6, 45))]   # [2026-10-17 removed the 0.5-weight panel]
    if standardized:      # each statistic z-scored across all scenarios x replicates of its design: three numbers on comparable ranges
        PANELS = [("z_rho", r"$s(|z_{\rho}|)$", None, None), ("z_C", r"$s(|z_{\rho_s}|)$", None, None), ("z_U", r"$s(|z_{\hat\rho_\Delta}|)$", None, None),
                  ("new3", r"$s(|z_\rho|) - s(|z_{\rho_s}|) + s(|z_{\hat\rho_\Delta}|)$", None, None)]
    if panels is not None: PANELS = panels
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 9.0), facecolor="none"); axs = axs.ravel()   # [2026-10-17 was 2x3 with a legend panel]
    for ax, (k, ptitle, thr, ylim) in zip(axs, PANELS):
        ax.set_facecolor("none")   # [2026-10-17 was symlog; linear axis requested]
        for i, sc in enumerate(order):
            for des, dx, alpha, hatched in DESIGNS:
                v = D[(D[scen_col] == sc) & (D.design == des)][k].dropna().values; c = COL[sc]
                b = ax.boxplot(v, positions=[i + dx], widths=0.34, patch_artist=True, showfliers=True, zorder=3, medianprops=dict(color=INK, lw=1.5),
                               whiskerprops=dict(color=c, lw=1.1), capprops=dict(color=c, lw=1.1), boxprops=dict(color=c, lw=1.3), flierprops=dict(marker="o", ms=2.6, mfc="none", mec=c, mew=0.8, alpha=.8))
                p = b["boxes"][0]; p.set_facecolor("white" if hatched else c)   # solid design = opaque scenario colour; hatched design = white background, scenario-coloured hatch (no alpha anywhere)   # [2026-10-17 was p.set_alpha(alpha): patch alpha also faded the hatch and the PDF dropped it]
                if hatched: p.set_hatch("////"); p.set_edgecolor(c); p.set_linewidth(1.3)
        if thr:
            for s_ in (thr, -thr): ax.axhline(s_, color=SUB, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=MUTED, lw=0.7, ls=":", zorder=1)
        ax.set_title(ptitle, loc="left", fontsize=11.5, color=INK, pad=8); ax.set_xticks(range(len(order))); ax.set_xticklabels(LAB, fontsize=8.5, color=SUB)
        vals = D[D.design.isin(designs)][k].dropna(); lo, hi = min(vals.min(), -(thr or 0)), max(vals.max(), thr or 0); pad = 0.05 * (hi - lo); ylim = (lo - pad, hi + pad)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10])); ax.tick_params(axis="y", labelsize=9, colors=SUB, length=3); ax.tick_params(axis="x", length=0)
        ax.set_ylim(*ylim); ax.set_xlim(-0.65, len(order) - 0.35); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
    h = [Patch(fc="#9aa0a6", ec=SUB, label="solid: " + legend_names[0]), Patch(fc="white", ec=SUB, hatch="////", label="hatched: " + legend_names[1])]
    fig.legend(handles=h, loc="upper right", ncol=2, frameon=False, fontsize=10.5, labelcolor=INK, handlelength=2.2, bbox_to_anchor=(0.995, 0.972))
    fig.text(0.012, 0.008, "100 replicates per box. Box = IQR, line = median, whiskers = 1.5 × IQR, circles = outliers." + ("" if (standardized or all(p[2] is None for p in PANELS)) else "  Dashed lines: single-pair threshold ±1.96 (z-score panels only).") + "\nSample = 20 h only; no cross-sample terms. Linear y axis (panel-specific range).\nBox colour = scenario (labelled on the x axis)." + note_extra, fontsize=8.6, color=SUB, va="bottom", ha="left", linespacing=1.45)
    fig.suptitle(title, x=0.012, ha="left", fontsize=12.5, color=INK, y=0.995); fig.tight_layout(rect=(0, 0.085, 1, 0.925)); os.makedirs(OUTDIR, exist_ok=True)
    for artist in fig.findobj():                                  # remove every clip/mask (Illustrator-editable PDF)
        if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
    fig.savefig(OUTDIR + outname + ".pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)          # transparent, editable PDF
    fig.savefig(OUTDIR + outname + ".png", dpi=170, bbox_inches="tight", facecolor=SURF)                                                       # opaque PNG preview only; plt.close(fig)
    print(outname); print(D.groupby(["design", scen_col])[[p[0] for p in PANELS]].median().round(2).to_string(), flush=True)
if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--random-pairs", action="store_true"); ap.add_argument("--standardize", action="store_true"); a = ap.parse_args()
    if not os.path.exists(CACHE):
        with mp.Pool(a.jobs) as p: res = p.map(run_rep, range(100), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(CACHE, index=False); print("computed ->", CACHE, flush=True)
    X = pd.read_csv(Z + "scenario_table_pictures_figure2.csv")[["scenario", "rep", "z_rho", "z_C", "z_U"]].assign(design="1500 twins + 3000 singletons")
    make_figure(pd.concat([X, pd.read_csv(CACHE)], ignore_index=True), ["1500 twins + 3000 singletons", "3000 twins"], "z-scores and combined scores: 1,500 twins + 3,000 singletons vs 3,000 twins", "zscore_boxplots_new_metrics_designs", ["1500 twins + 3000 singletons", "3000 twins"])
    if a.standardize:
        Ds = pd.concat([X, pd.read_csv(CACHE)], ignore_index=True)
        for k in ("z_rho", "z_C", "z_U"):      # s(|z|) = (|z| - mean) / sd(ddof=1) over all 600 rows (6 scenarios x 100 replicates) of the design, as todo4v2.s() does per panel
            Ds[k] = Ds.groupby("design")[k].transform(lambda v: (v.abs() - v.abs().mean()) / v.abs().std(ddof=1))
        Ds["new3"] = Ds.z_rho - Ds.z_C + Ds.z_U
        make_figure(Ds, ["1500 twins + 3000 singletons", "3000 twins"], "Standardized statistics: each |z| z-scored across all scenarios and replicates of its design", "zscore_boxplots_new_metrics_designs_standardized", ["1500 twins + 3000 singletons", "3000 twins"],
                    "\ns(·) = (|z| − mean) / sd over all 600 rows (6 scenarios × 100 replicates) of the same design, so the three terms of the metric have comparable ranges.", standardized=True)
    if a.random_pairs:
        if not os.path.exists(CACHE_R):
            with mp.Pool(a.jobs) as p: res = p.map(run_rep_random, range(100), chunksize=1)
            pd.DataFrame([r for rr in res for r in rr]).to_csv(CACHE_R, index=False); print("computed ->", CACHE_R, flush=True)
        make_figure(pd.read_csv(CACHE_R), ["1500 random pairs + 3000 singletons", "3000 random pairs"], "Control: random cell pairs used as twins (sister pairing broken; same cells and designs)", "zscore_boxplots_new_metrics_random_pairs",
                    ["1500 random pairs + 3000 singletons", "3000 random pairs"], "\nRandom pairs: each clone's second cell is swapped with the second cell of another clone chosen from all 6,000 clones, so no pair shares a clone\nand in the 2-state / drift scenarios about half of the pairs combine a low- and a high-k_on cell (measured: 50.1%).")
