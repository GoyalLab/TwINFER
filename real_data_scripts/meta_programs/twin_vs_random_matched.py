#!/usr/bin/env python3
"""Expression-matched version of the twin-vs-random fold-change comparison.

Input : analysis_data/<fm>/data/twin_vs_random_logfc_top50/gene_table_all.csv  (every gene detected in >= 5 % of cells; made by twin_vs_random_logfc.py --allgenes)
        per gene: twin_mean_abs_log2FC, random_mean_abs_log2FC (clone-mate pairs vs random pairs), mean_expr = mean log2(CP10k/10 + 1), group (program group or 'background')
Matching: for every program gene the --nctrl (default 100) non-program genes with the closest mean expression are its controls (ties in mean_expr broken by detection %).
          The statistic is log2(twin / random). excess = gene's log2 ratio minus the median log2 ratio of its controls (negative = more twin-similar than expression-matched genes).
          empirical p = (1 + number of controls with a log2 ratio <= the gene's) / (nctrl + 1)  (one-sided, 'more similar in twins than matched genes').
Group level: median excess, share of genes with empirical p < 0.05, one-sided Wilcoxon signed-rank test of the excess against 0 (genes of one program are correlated, so the p-value is optimistic).
Outputs: gene_matched_table.csv, group_matched_summary.csv, matched.pdf (Arial, fonttype 42, no clip paths, transparent).   usage: FM_DATASET=FM01 twin_vs_random_matched.py [--nctrl 100]
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; DS = os.environ.get("FM_DATASET", "FM01")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import twin_vs_random_logfc as W
SRC = os.environ.get("PROG_SRC", "mp_nmf"); SUF = "_pooled" if SRC == "mp_nmf_pooled" else ""
PAL = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#7B3294"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nctrl", type=int, default=100); a = ap.parse_args()
    base = f"{ROOT}/analysis_data/{DS.lower()}/data/twin_vs_random_logfc_top50"; out = base + SUF; os.makedirs(out, exist_ok=True); T = pd.read_csv(f"{base}/gene_table_all.csv")
    T["log2r"] = np.log2(T.twin_mean_abs_log2FC / T.random_mean_abs_log2FC); T = T[np.isfinite(T.log2r)].reset_index(drop=True)
    GR = W.build_groups(DS, 50, SRC); gset = {g for v in GR.values() for g in v}   # program groups from the chosen source; a gene in several programs is evaluated once per program
    C = T[~T.gene.isin(gset)].reset_index(drop=True); cx, cd, cr = C.mean_expr.to_numpy(), C.pct_cells_detected.to_numpy(), C.log2r.to_numpy(); Ti = T.set_index("gene")
    rows = []
    for n, genes_ in GR.items():
        for gname in genes_:
            if gname not in Ti.index: continue
            g = Ti.loc[gname]; dist = np.abs(cx - g.mean_expr) + 1e-6 * np.abs(cd - g.pct_cells_detected); j = np.argpartition(dist, a.nctrl)[:a.nctrl]
            ctrl = cr[j]; rows.append(dict(gene=gname, group=n, mean_expr=g.mean_expr, pct_detected=g.pct_cells_detected, twin=g.twin_mean_abs_log2FC, random=g.random_mean_abs_log2FC, log2r=g.log2r,
                                           ctrl_median_log2r=np.median(ctrl), ctrl_mean_expr_range=f"{cx[j].min():.3f}-{cx[j].max():.3f}", excess=g.log2r - np.median(ctrl), emp_p=(1 + (ctrl <= g.log2r).sum()) / (a.nctrl + 1)))
    G = pd.DataFrame(rows); G.to_csv(f"{out}/gene_matched_table.csv", index=False)
    # same exercise for a random set of non-program genes (null for the group summaries): each treated as if it were a program gene, its own controls drawn from the other non-program genes
    rng = np.random.default_rng(0); null = []
    for i in rng.choice(len(C), min(600, len(C)), replace=False):
        dist = np.abs(cx - cx[i]) + 1e-6 * np.abs(cd - cd[i]); dist[i] = np.inf; j = np.argpartition(dist, a.nctrl)[:a.nctrl]; null.append((cr[i] - np.median(cr[j]), (1 + (cr[j] <= cr[i]).sum()) / (a.nctrl + 1)))
    null = np.array(null)
    S = []
    for n, d in G.groupby("group"):
        w = wilcoxon(d.excess, alternative="less") if len(d) > 5 and (d.excess != 0).any() else None
        S.append(dict(dataset=DS, group=n, n_genes=len(d), median_excess_log2r=d.excess.median(), median_raw_log2r=d.log2r.median(), frac_emp_p_lt_0_05=float((d.emp_p < 0.05).mean()),
                      frac_below_matched_median=float((d.excess < 0).mean()), wilcoxon_p_less=float(w.pvalue) if w else np.nan))
    S.append(dict(dataset=DS, group="random non-program genes (null)", n_genes=len(null), median_excess_log2r=float(np.median(null[:, 0])), median_raw_log2r=float(C.log2r.median()), frac_emp_p_lt_0_05=float((null[:, 1] < 0.05).mean()), frac_below_matched_median=float((null[:, 0] < 0).mean()), wilcoxon_p_less=float(wilcoxon(null[:, 0], alternative="less").pvalue)))
    S = pd.DataFrame(S); S.to_csv(f"{out}/group_matched_summary.csv", index=False)
    pd.set_option("display.width", 220); print(S.round(4).to_string(index=False)); print("max expression gap between a gene and its controls (median over genes):", round(float(np.median([max(abs(float(x) - m) for x in r.split("-")) for r, m in zip(G.ctrl_mean_expr_range, G.mean_expr)])), 4))
    plot(T, C, G, out)


def plot(T, C, G, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import font_manager as fm
    fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.fontset": "cm",
                         "axes.labelsize": 15, "axes.titlesize": 15, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10})
    names = list(dict.fromkeys(G.group)); col = {n: PAL[k % len(PAL)] for k, n in enumerate(names)}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    A = ax[0]; o = np.argsort(C.mean_expr.to_numpy()); x = C.mean_expr.to_numpy()[o]; y = C.log2r.to_numpy()[o]
    A.scatter(x, y, s=3, c="#CFCFCF", lw=0, rasterized=False, zorder=1, label=f"all other detected genes (n={len(C)})")
    w = max(100, len(x) // 40); ctr = np.arange(w // 2, len(x) - w // 2, w // 4)
    mid = [np.median(y[c - w // 2:c + w // 2]) for c in ctr]; lo = [np.percentile(y[c - w // 2:c + w // 2], 2.5) for c in ctr]; hi = [np.percentile(y[c - w // 2:c + w // 2], 97.5) for c in ctr]; xc = [np.median(x[c - w // 2:c + w // 2]) for c in ctr]
    A.fill_between(xc, lo, hi, color="#7A7A7A", alpha=0.25, lw=0, zorder=2, label="matched-gene 2.5 to 97.5 %"); A.plot(xc, mid, color="#333333", lw=1.8, zorder=3, label="median of all genes at that expression")
    for n in names:
        d = G[G.group == n]; A.scatter(d.mean_expr, d.log2r, s=28, c=col[n], edgecolors="black", linewidths=0.4, zorder=4, label=f"{n} (n={len(d)})")
    A.axhline(0, color="#444444", lw=1, ls="--"); A.set_xscale("log"); A.set_xlabel("Mean expression, log$_2$(CP10k/10 + 1) (log scale)"); A.set_ylabel("log$_2$(twin / random) mean |log$_2$ FC|")
    A.legend(frameon=False, fontsize=8.5, loc="lower right")
    B = ax[1]; rng = np.random.default_rng(1)
    for k, n in enumerate(names):
        d = G[G.group == n]; B.scatter(k + rng.uniform(-0.25, 0.25, len(d)), d.excess, s=16, c=col[n], edgecolors="black", linewidths=0.3, zorder=3)
        B.plot([k - 0.35, k + 0.35], [d.excess.median()] * 2, color="black", lw=2, zorder=4)
    B.axhline(0, color="#444444", lw=1, ls="--"); B.set_xticks(range(len(names))); B.set_xticklabels([n.replace("Programs ", "").replace("Program ", "") for n in names], rotation=0)
    B.set_xlabel("Program (A / B replicate pair)"); B.set_ylabel("Excess over expression-matched genes\n(log$_2$ ratio $-$ median of 100 controls)")
    for a_ in ax:
        a_.set_facecolor("none")
        for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
    fig.tight_layout()
    for art in fig.findobj():
        if hasattr(art, "set_clip_on"): art.set_clip_on(False)
    fig.savefig(f"{out}/matched.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True); print("saved", f"{out}/matched.pdf")


if __name__ == "__main__":
    main()
