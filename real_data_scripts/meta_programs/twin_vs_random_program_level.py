#!/usr/bin/env python3
"""Program-level, clone-level, expression-matched test: are the genes of a program, taken together, more similar in twins than in random pairs?

Per gene g and clone c (from twin_vs_random_logfc.py --allgenes): delta_c(g) = d_c(g) - r_c(g), d = mean |log2 FC| between the clone's cell pairs, r = the same for randomly relabelled cells.
Program statistic for clone c: S_c = mean over the program's genes of delta_c(g)   (one value per program per clone; cells of one clone are one observation, genes are averaged first,
so the correlation between the program's genes is carried inside S_c).
 1. clone-level test: one-sided paired t-test of mean_c S_c < 0 across the N_t clones (and a Wilcoxon signed-rank test)
 2. expression-matched controls: --nsets (500) control gene sets, each program gene replaced by one gene drawn at random from its --nctrl (100) non-program genes closest in mean expression;
    empirical p = (1 + #{control sets with mean_c S_c <= the program's}) / (nsets + 1); z = (program - mean of controls) / sd of controls
 3. gene correlation: mean pairwise correlation across clones of the genes' delta_c, for the program and for the control sets, and the implied effective number of independent genes n_eff = n / (1 + (n - 1) rho)
Outputs: analysis_data/<fm>/data/twin_vs_random_logfc_top50/program_level_summary.csv     usage: FM_DATASET=FM01 twin_vs_random_program_level.py
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import t as tdist, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; DS = os.environ.get("FM_DATASET", "FM01")
import twin_vs_random_logfc as W
SRC = os.environ.get("PROG_SRC", "mp_nmf"); SUF = "_pooled" if SRC == "mp_nmf_pooled" else ""


def mean_corr(M):
    """mean pairwise Pearson correlation between the columns of M (clones x genes)"""
    if M.shape[1] < 2: return np.nan
    C = np.corrcoef(M, rowvar=False); n = C.shape[0]; return (C.sum() - n) / (n * (n - 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nsets", type=int, default=500); ap.add_argument("--nctrl", type=int, default=100); a = ap.parse_args(); rng = np.random.default_rng(0)
    out = f"{ROOT}/analysis_data/{DS.lower()}/data/twin_vs_random_logfc_top50"
    Z = np.load(f"{out}/delta_clone_gene.npz", allow_pickle=True); dl = Z["delta"]; gn = Z["genes"]; gi = {g: i for i, g in enumerate(gn)}; Nt = dl.shape[0]
    T = pd.read_csv(f"{out}/gene_table_all.csv").set_index("gene").loc[gn]; mx = T.mean_expr.to_numpy(); pct = T.pct_cells_detected.to_numpy(); rnd = T.random_mean_abs_log2FC.to_numpy()
    G = W.build_groups(DS, 50, SRC); gset = {g for v in G.values() for g in v}
    prog_idx = {n: np.array([gi[g] for g in v if g in gi]) for n, v in G.items()}; prog_idx = {n: i for n, i in prog_idx.items() if len(i) >= 10}; ctrl = np.array([i for i, g in enumerate(gn) if g not in gset])
    rows = []
    for n, idx in prog_idx.items():
        S = dl[:, idx].mean(1); m = S.mean(); se = S.std(ddof=1) / np.sqrt(Nt); t = m / se; p_t = tdist.cdf(t, Nt - 1)
        w = wilcoxon(S, alternative="less").pvalue
        cand = []
        for i in idx:
            dist = np.abs(mx[ctrl] - mx[i]) + 1e-6 * np.abs(pct[ctrl] - pct[i]); cand.append(ctrl[np.argpartition(dist, a.nctrl)[:a.nctrl]])
        mc = np.empty(a.nsets); rc = np.empty(a.nsets); rho_c = []
        for k in range(a.nsets):
            sel = np.array([rng.choice(c) for c in cand]); mc[k] = dl[:, sel].mean(); rc[k] = mc[k] / rnd[sel].mean()
            if k < 50: rho_c.append(mean_corr(dl[:, sel]))
        rho = mean_corr(dl[:, idx]); n_g = len(idx); neff = n_g / (1 + (n_g - 1) * rho) if rho > 0 else float(n_g); rho_ctrl = float(np.nanmean(rho_c)); neff_c = n_g / (1 + (n_g - 1) * rho_ctrl) if rho_ctrl > 0 else float(n_g)
        rows.append(dict(dataset=DS, group=n, n_genes=n_g, N_clones=Nt, mean_delta_program=m, mean_delta_matched_controls=mc.mean(), sd_matched_controls=mc.std(ddof=1), z_vs_matched=(m - mc.mean()) / mc.std(ddof=1),
                         emp_p_vs_matched=(1 + (mc <= m).sum()) / (a.nsets + 1), clone_level_t=t, clone_level_p_t=p_t, clone_level_p_wilcoxon=w,
                         rel_reduction_program=m / rnd[idx].mean(), rel_reduction_matched_controls=rc.mean(), mean_gene_gene_corr_program=rho, mean_gene_gene_corr_controls=rho_ctrl, n_eff_program=neff, n_eff_controls=neff_c))
    S = pd.DataFrame(rows); S.to_csv(f"{out}/program_level_summary{SUF}.csv", index=False); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.drop(columns=["dataset", "N_clones"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
