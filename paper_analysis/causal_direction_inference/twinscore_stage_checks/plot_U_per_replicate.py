"""Per-replicate unshared correlation U (= S - C, twin-centred rank units) for the two SINGLE-STATE scenarios (A,B: no regulation; A->B: regulation), with the shared part C for comparison.
Designs: 1,500 twins + 3,000 singletons (existing table scenario_table_pictures_figure2.csv: S, C, U) and 3,000 twins (computed here from the same simulation files, same random clone subset as plot_new_metrics_boxplots.py).
Output: analysis_data/paper_analysis/final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/U_per_replicate_single_state.{pdf,png};  cache <O>/U_C_S_3000twins_single_state.csv"""
import argparse, os, numpy as np, pandas as pd
from figure2_1000_zscores import NCL, G, find
from motif_shared_driver import zC
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/"; O = A + "causal_direction_inference/zrho_pdf_network_auprc/"; Z = A + "causal_direction_inference/figure2_1000_zscores/"
OUTDIR = A + "final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; CACHE = O + "U_C_S_3000twins_single_state.csv"; SC = [("A_B", "A_B (single state)", 0), ("A_to_B", "A_to_B (single state)", 1)]
def one(rep):
    rows = []
    for cfg, name, si in SC:
        d = pd.read_csv(find(cfg, rep)[0], usecols=["time_step", "gene_1_mRNA", "gene_2_mRNA", "replicate", "clone_id"]); s = d[d.time_step == 20]
        P = {g: s.pivot(index="clone_id", columns="replicate", values=f"{g}_mRNA").values for g in G}
        c = np.random.RandomState(rep * 100 + si + 12345).permutation(NCL)[:3000]; S_, C_, zc = zC(P["gene_1"][c], P["gene_2"][c])
        rows.append(dict(scenario=name, rep=rep, design="3000 twins", S=S_, C=C_, U=S_ - C_))
    return rows
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    if not os.path.exists(CACHE):
        import multiprocessing as mp
        with mp.Pool(a.jobs) as p: res = p.map(one, range(100), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(CACHE, index=False)
    X = pd.read_csv(Z + "scenario_table_pictures_figure2.csv"); X = X[X.scenario.str.contains("single")][["scenario", "rep", "S", "C", "U"]].assign(design="1500 twins + 3000 singletons")
    D = pd.concat([X, pd.read_csv(CACHE)], ignore_index=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import font_manager as fm; from matplotlib.lines import Line2D
    for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.sans-serif": ["Arial"], "font.family": "sans-serif", "mathtext.fontset": "cm",
                         "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})   # [2026-10-17 user's Illustrator-editable recipe, memory feedback_illustrator_editable_plots]
    INK, SUB, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"; COL = {"A_B (single state)": "#2a78d6", "A_to_B (single state)": "#eb6834"}
    ROWS = [("A_B (single state)", "A, B  (no regulation)"), ("A_to_B (single state)", "A→B  (regulation)")]; COLS = ["1500 twins + 3000 singletons", "3000 twins"]
    lo, hi = D[["U", "C"]].min().min(), D[["U", "C"]].max().max(); pad = 0.06 * (hi - lo)
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 8.4), facecolor="none", sharey=True, sharex=True)
    for i, (sc, lab) in enumerate(ROWS):
        for j, des in enumerate(COLS):
            ax = axs[i, j]; ax.set_facecolor("none"); x = D[(D.scenario == sc) & (D.design == des)].sort_values("rep")
            ax.axhline(0, color=MUTED, lw=0.8, ls=":", zorder=1)
            ax.scatter(x.rep, x.C, s=16, facecolors="none", edgecolors=SUB, linewidths=0.8, zorder=2); ax.scatter(x.rep, x.U, s=17, color=COL[sc], alpha=0.85, linewidths=0, zorder=3)
            ax.axhline(x.C.median(), color=SUB, lw=1.2, ls=(0, (4, 3)), zorder=2); ax.axhline(x.U.median(), color=COL[sc], lw=1.6, zorder=2)
            frac = (x.U.abs() > x.C.abs()).mean() * 100; share = (x.U / x.S).median() if abs(x.S.median()) > 0.01 else np.nan   # [2026-10-17 was median(U) / median(S)]
            txt = f"median U = {x.U.median():+.4f}\\nmedian C = {x.C.median():+.4f}\\n|U| > |C| in {frac:.0f}% of replicates" + (f"\\nmedian (U / S) = {share:.2f}" if np.isfinite(share) else "")
            ax.text(0.015, 0.975, txt.replace("\\n", "\n"), transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK, linespacing=1.4)
            ax.set_title(f"{lab} — {des}", loc="left", fontsize=11, color=INK, pad=7); ax.set_ylim(lo - pad, hi + 0.075)   # headroom for the annotation; ax.set_xlim(-2, 101); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
            for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
            ax.tick_params(colors=SUB, labelsize=9, length=3)
            if j == 0: ax.set_ylabel("correlation (twin-centred ranks)", fontsize=9.5, color=SUB)
            if i == 1: ax.set_xlabel("replicate", fontsize=9.5, color=SUB)
    h = [Line2D([], [], marker="o", ls="", color="#2a78d6", label="U (unshared)  — A,B blue / A→B orange"), Line2D([], [], marker="o", ls="", mfc="none", mec=SUB, label="C (shared)"), Line2D([], [], color=SUB, ls=(0, (4, 3)), label="medians")]
    fig.legend(handles=h, loc="upper right", ncol=3, frameon=False, fontsize=10, labelcolor=INK, bbox_to_anchor=(0.995, 0.972))
    fig.suptitle("Unshared (U) vs shared (C) correlation per replicate, single-state scenarios", x=0.012, ha="left", fontsize=12.5, color=INK, y=0.995)
    fig.text(0.012, 0.008, "One point per replicate (100). U = S − C: same-cell correlation minus cross-sister correlation (twin-centred ranks); S ≈ ρ in a single state.\nSample = 20 h only; no cross-sample terms. Shared y axis across the four panels.", fontsize=8.6, color=SUB, va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.05, 1, 0.925)); os.makedirs(OUTDIR, exist_ok=True)
    for artist in fig.findobj():                                  # remove every clip/mask (Illustrator-editable PDF)
        if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
    fig.savefig(OUTDIR + "U_per_replicate_single_state.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)          # transparent, editable PDF
    fig.savefig(OUTDIR + "U_per_replicate_single_state.png", dpi=170, bbox_inches="tight", facecolor=SURF)                                                       # opaque PNG preview only
    t = D.groupby(["scenario", "design"]).agg(med_S=("S", "median"), med_C=("C", "median"), med_U=("U", "median"), U_gt_C=("U", lambda v: np.nan)).drop(columns="U_gt_C"); t["median_U_over_S_per_rep"] = D.assign(r=D.U / D.S).groupby(["scenario", "design"]).r.median(); t["IQR_U_over_S"] = D.assign(r=D.U / D.S).groupby(["scenario", "design"]).r.apply(lambda v: v.quantile(.75) - v.quantile(.25)); t["ratio_of_medians"] = t.med_U / t.med_S
    t["pct_absU_gt_absC"] = D.assign(f=D.U.abs() > D.C.abs()).groupby(["scenario", "design"]).f.mean() * 100; print(t.round(4).to_string())
