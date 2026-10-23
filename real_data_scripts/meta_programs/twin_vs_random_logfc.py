#!/usr/bin/env python3
"""Per-gene similarity of clone mates (twins) vs random cell pairs, as the average absolute log fold change between the two cells.

For every gene g (log2(CP10k/10 + 1) expression x):
  twin value   d_c(g) = mean over the cell pairs of clone c of |x_a - x_b|   (|log2 fold change| between the two cells; every within-clone pair, weight 1/P_c per pair = clone mean)
  random value r_c(g) = the same pairs with the cells replaced by random cells (the cell labels permuted over the whole dataset: same clone sizes, no clonal relation), averaged over --nperm permutations
  plotted: mean_c d_c (twin) against mean_c r_c (random); below the diagonal = more similar in twins
  test:    one-sided paired t-test over the N_t clones of delta_c = d_c - r_c  (H1: twins have a smaller fold change than random pairs); BH over all tested genes
Genes: --nbg random genes from those detected in >= 5 % of cells (seed 0) as the background, plus the top-20 genes of each highlighted program.
Highlighted programs: the robust programs found in both replicates (greedy A/B pairing, >= 35 shared of 50 genes; all programs for a one-replicate dataset); each group = union of the top --topn genes of the two programs; names = global program numbers.
usage: FM_DATASET=FM01 twin_vs_random_logfc.py [--nbg 1000] [--nperm 5] [--topn 50]   (output folder twin_vs_random_logfc_top50; --topn 20 writes twin_vs_random_logfc)
Outputs: analysis_data/<fm>/data/twin_vs_random_logfc/{gene_table.csv, scatter.pdf, summary.json}
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import t as tdist

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; DS = os.environ.get("FM_DATASET", "FM01"); QC = f"{ROOT}/finalized_data/{DS}_data/qc_filtered"
# GROUPS (FM01 top-20 version, kept for reference; replaced by build_groups below)
# GROUPS = {"Programs 1 / 8": (("A", 4, 3), ("B", 4, 3)), "Programs 2 / 7": (("A", 6, 1), ("B", 4, 1)), "Programs 3 / 9": (("A", 8, 2), ("B", 6, 2)), "Programs 5 / 6": (("A", 9, 0), ("B", 4, 0))}
PAL = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#7B3294"]


def build_groups(ds, topn, src=None):
    """Highlight groups = robust programs found in both replicates: greedy pairing of the programs of the two replicates by top-50 overlap (>= 35 genes); group = union of the
    two programs' top-n genes. A dataset with one replicate (FM03) uses every robust program as its own group. Names use the global program numbers of program_index.csv."""
    src = src or os.environ.get("PROG_SRC", "mp_nmf")   # "mp_nmf": replicate-A/B pairing (split halves of one experiment, NOT independent replicates); "mp_nmf_pooled": A+B merged, one sample, each robust program its own group
    R = pd.read_csv(f"{ROOT}/analysis_data/{ds.lower()}/data/{src}/robust_programs.csv"); PI = pd.read_csv(f"{ROOT}/analysis_data/fm_meta_programs_comparison{'_pooled' if src == 'mp_nmf_pooled' else ''}/program_index.csv")
    PI = PI[PI.dataset == ds].set_index(["replicate", "K", "idx"]).program
    R["num"] = [int(PI.loc[(r.sample, r.K, r.idx)]) for r in R.itertuples()]; reps = sorted(R["sample"].unique())
    G = {}
    if len(reps) < 2:   # one sample (pooled mode, or FM03)
        for r in R.itertuples(): G[f"Program {r.num}"] = r.genes.split()[:topn]
        return G
    A, B = R[R["sample"] == reps[0]], R[R["sample"] == reps[1]]
    cand = sorted([(len(set(a.genes.split()) & set(b.genes.split())), a.num, b.num) for a in A.itertuples() for b in B.itertuples()], reverse=True); ua, ub = set(), set()
    for ov, na, nb in cand:
        if ov < 35 or na in ua or nb in ub: continue
        ua.add(na); ub.add(nb); g = R[R.num == na].genes.iloc[0].split()[:topn] + R[R.num == nb].genes.iloc[0].split()[:topn]; G[f"Programs {na} / {nb}"] = list(dict.fromkeys(g))
    return G


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty_like(q); r[o] = np.minimum(q, 1); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nbg", type=int, default=1000); ap.add_argument("--nperm", type=int, default=5); ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--max-pairs", type=int, default=0, help="sample at most this many cell pairs per clone (0 = all); the clone value is a mean over pairs, so sampling only adds noise (needed for giant clones, e.g. FM08)")
    ap.add_argument("--allgenes", action="store_true", help="test every gene detected in >= 5 %% of cells (for expression matching); writes gene_table_all.csv, no plot")
    a = ap.parse_args(); rng = np.random.default_rng(0)
    out = f"{ROOT}/analysis_data/{DS.lower()}/data/twin_vs_random_logfc" + ("" if a.topn == 20 else f"_top{a.topn}") + ("_pooled" if (os.environ.get("PROG_SRC") == "mp_nmf_pooled" and not a.allgenes) else ""); os.makedirs(out, exist_ok=True)
    genes = np.array([l.strip() for l in open(f"{QC}/genes.txt")]); gi = {g: i for i, g in enumerate(genes)}
    cells = np.array([l.strip() for l in open(f"{QC}/cells.txt")]); obs = pd.read_csv(f"{QC}/obs_metadata.csv", index_col=0).loc[cells]
    X = sio.mmread(f"{QC}/{DS.lower()}_qc_counts.mtx").tocsr().astype(np.float32); tot = np.asarray(X.sum(1)).ravel()
    X = X.multiply(1e4 / tot[:, None]).tocsc(); X.data = np.log2(X.data / 10 + 1); det = np.asarray((X > 0).mean(0)).ravel()
    # twin pairs, contiguous per clone
    cl = obs["fatemap_clone_singletcode"]; a_, b_, start, P = [], [], [], []; pos = 0
    for c, idx in pd.Series(np.arange(len(cl))).groupby(cl.to_numpy()).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2: continue
        ia, ib = np.triu_indices(len(idx), 1)
        if a.max_pairs and len(ia) > a.max_pairs: sel = np.sort(rng.choice(len(ia), a.max_pairs, replace=False)); ia, ib = ia[sel], ib[sel]
        a_.append(idx[ia]); b_.append(idx[ib]); start.append(pos); P.append(len(ia)); pos += len(ia)
    pa, pb, start, P = np.concatenate(a_), np.concatenate(b_), np.array(start), np.array(P, float); Nt = len(P)
    print(DS, "cells", len(cells), "clones with >=2 cells", Nt, "twin pairs", len(pa), flush=True)
    # programs
    GROUPS = build_groups(DS, a.topn); grp = {n: [x for x in v if x in gi and det[gi[x]] >= 0.05] for n, v in GROUPS.items()}
    COL = {n: PAL[k % len(PAL)] for k, n in enumerate(GROUPS)}
    pool = np.where(det >= 0.05)[0]; bg = list(rng.choice(pool, a.nbg, replace=False))
    allg = list(dict.fromkeys(bg + [gi[x] for v in grp.values() for x in v]))
    if a.allgenes: allg = list(dict.fromkeys([int(g) for g in pool] + [gi[x] for v in grp.values() for x in v])); print("genes tested", len(allg), flush=True)
    perms = [rng.permutation(len(cells)) for _ in range(a.nperm)]
    rows = []; DL = []
    cs = int(np.clip(2.0e7 / len(pa), 4, 100))   # genes per chunk, bounded by memory (many pairs in FM08)
    for s in range(0, len(allg), cs):
        ch = allg[s:s + cs]; D = X[:, ch].toarray()
        d = np.add.reduceat(np.abs(D[pa] - D[pb]), start, axis=0) / P[:, None]
        r = np.mean([np.add.reduceat(np.abs(D[p[pa]] - D[p[pb]]), start, axis=0) / P[:, None] for p in perms], axis=0)
        dl = d - r; DL.append(dl.astype(np.float32)); m = dl.mean(0); se = dl.std(0, ddof=1) / np.sqrt(Nt); tt = m / se
        for k, g in enumerate(ch):
            rows.append(dict(gene=genes[g], twin_mean_abs_log2FC=d[:, k].mean(), random_mean_abs_log2FC=r[:, k].mean(), mean_diff=m[k], t=tt[k], p_one_sided=tdist.cdf(tt[k], Nt - 1), pct_cells_detected=100 * det[g], mean_expr=float(D[:, k].mean())))
    T = pd.DataFrame(rows); T["log2_twin_over_random"] = np.log2(T.twin_mean_abs_log2FC / T.random_mean_abs_log2FC); T["q"] = bh(T.p_one_sided)
    T["group"] = "background"
    gl = {x: n for n, v in grp.items() for x in v}   # a gene in two groups keeps the first
    T.loc[T.gene.isin(gl), "group"] = T.gene.map(gl)
    T["in_background_sample"] = T.gene.isin(set(genes[bg]))
    T.to_csv(f"{out}/gene_table{'_all' if a.allgenes else ''}.csv", index=False)
    if a.allgenes:
        np.savez_compressed(f"{out}/delta_clone_gene.npz", delta=np.hstack(DL), genes=genes[allg]); print("saved all-gene table", len(T), "and per-clone deltas", np.hstack(DL).shape); return
    B = T[T.in_background_sample]
    summ = dict(dataset=DS, N_t_clones=Nt, n_twin_pairs=int(len(pa)), nperm=a.nperm, n_background=len(B), background_sig_q05=int((B.q < 0.05).sum()), background_frac_below_diagonal=float((B.mean_diff < 0).mean()),
                background_median_log2_ratio=float(B.log2_twin_over_random.median()), program_genes={n: len(v) for n, v in grp.items()})
    for n in GROUPS:
        G = T[T.group == n]; summ[n] = dict(n=len(G), sig_q05=int((G.q < 0.05).sum()), frac_below_diagonal=float((G.mean_diff < 0).mean()), median_log2_ratio=float(G.log2_twin_over_random.median()))
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1); print(json.dumps(summ, indent=1), flush=True)
    plot(T, out, COL, a.topn)


def plot(T, out, COL, topn):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import font_manager as fm
    fp = f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf"; fm.fontManager.addfont(fp)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.fontset": "cm",
                         "axes.labelsize": 16, "axes.titlesize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.8))
    B = T[(T.group == "background") & T.in_background_sample]; sig = B.q < 0.05
    lo = min(T.random_mean_abs_log2FC.min(), T.twin_mean_abs_log2FC.min()) * 0.9; hi = max(T.random_mean_abs_log2FC.max(), T.twin_mean_abs_log2FC.max()) * 1.05
    for i, (xc, yc, xl, yl) in enumerate([("random_mean_abs_log2FC", "twin_mean_abs_log2FC", "Random pairs, mean |log$_2$ FC|", "Twins (clone mates), mean |log$_2$ FC|"), ("log2_twin_over_random", None, "log$_2$(twin / random)", "$-$log$_{10}$ q (one-sided)")]):
        A = ax[i]
        if i == 0:
            A.plot([lo, hi], [lo, hi], color="#444444", lw=1, ls="--", zorder=1); A.set_xlim(lo, hi); A.set_ylim(lo, hi)
            A.scatter(B[xc][~sig], B[yc][~sig], s=9, c="#CFCFCF", lw=0, label=f"background genes, q $\\geq$ 0.05 (n={int((~sig).sum())})", zorder=2)
            A.scatter(B[xc][sig], B[yc][sig], s=9, c="#7A7A7A", lw=0, label=f"background genes, more similar in twins, q < 0.05 (n={int(sig.sum())})", zorder=3)
        else:
            y = -np.log10(T.q.clip(lower=1e-300)); yb = -np.log10(B.q.clip(lower=1e-300))
            A.scatter(B[xc][~sig], yb[~sig], s=9, c="#CFCFCF", lw=0, zorder=2); A.scatter(B[xc][sig], yb[sig], s=9, c="#7A7A7A", lw=0, zorder=3)
            A.axhline(-np.log10(0.05), color="#444444", lw=1, ls=":"); A.axvline(0, color="#444444", lw=1, ls="--")
        for n, c in COL.items():
            G = T[T.group == n]
            if i == 0: A.scatter(G[xc], G[yc], s=34, c=c, edgecolors="black", linewidths=0.4, label=f"{n} (top {topn} genes, n={len(G)})", zorder=4)
            else: A.scatter(G[xc], -np.log10(G.q.clip(lower=1e-300)), s=34, c=c, edgecolors="black", linewidths=0.4, zorder=4)
        A.set_xlabel(xl); A.set_ylabel(yl); A.set_facecolor("none")
        for s_ in ("top", "right"): A.spines[s_].set_visible(False)
    ax[0].legend(frameon=False, loc="upper left", fontsize=9.5); fig.tight_layout()
    for art in fig.findobj():
        if hasattr(art, "set_clip_on"): art.set_clip_on(False)
    fig.savefig(f"{out}/scatter.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    print("saved", f"{out}/scatter.pdf")


if __name__ == "__main__":
    main()
