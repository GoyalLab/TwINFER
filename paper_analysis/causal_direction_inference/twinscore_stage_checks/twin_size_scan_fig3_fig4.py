"""Twin-count scan for the greyscale by-network plots: (a) figure 3 conditions at N = 6000 (all clones; 250-3000 already in twin_size_scan_figure3.csv, same clone permutation RandomState(rep*100+si+12345) -> nested);
(b) figure 4 motifs (Fan_out, Feed_forward, Mutual_regulation; 1k_per_motif_v2; first --nreps replicates finished for ALL motifs) at N = 250, 500, 1000, 3000, 6000 for pairs 1-2, 1-3, 2-3 (permutation RandomState(rep).permutation(6000), nested).
z_rho, z_C, z_U etc. from twin_decomposition.twin_stats on the 20 h sister pairs.  Output: <O>/twin_size_scan_figure3_6000.csv, <O>/twin_size_scan_figure4.csv   usage: python twin_size_scan_fig3_fig4.py [--jobs 8] [--nreps 100]"""
import argparse, glob, os, re, numpy as np, pandas as pd
from twin_decomposition import twin_stats
import figure3_decomposition as F3
from figure2_1000_zscores import load as load3, NCL
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; SIM4 = A + "simulation_data/figure_4/1k_per_motif_v2/"
MOT = ["Fan_out", "Feed_forward", "Mutual_regulation"]; PAIRS = [(1, 2), (1, 3), (2, 3)]; NS4 = (250, 500, 1000, 3000, 6000)
def fig3_rep(rep):
    rows = []
    for si, cond in enumerate(F3.CONDS):
        s = load3(F3.find(cond, F3.REPS[cond][rep])); P2 = s[20]; cc = np.random.RandomState(rep * 100 + si + 12345).permutation(NCL)
        t = twin_stats(P2["gene_1"][cc[:6000]], P2["gene_2"][cc[:6000]]); t.update(condition=cond, rep=rep, N=6000); rows.append(t)
    return rows
def files4():
    out = {}
    for c, m in enumerate(MOT):
        for f in glob.glob(SIM4 + f"df_rows_*_ncells_6000_{m}_{c}_*_*.csv"):
            out[(m, int(re.search(rf"_{m}_{c}_(\d+)_[0-9a-f]{{8}}\.csv$", f).group(1)))] = f      # last file wins if a replicate id is duplicated (same rule as motif_replicates_zscores.py)
    return out
def fig4_rep(a):
    m, r, f = a; d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate", "gene_1_mRNA", "gene_2_mRNA", "gene_3_mRNA"]); d = d[d.time_step == 20]
    X = {g: d.pivot(index="clone_id", columns="replicate", values=f"gene_{g}_mRNA").values for g in (1, 2, 3)}; perm = np.random.RandomState(r).permutation(X[1].shape[0]); rows = []
    for N in NS4:
        c = perm[:N]
        for i, j in PAIRS:
            t = twin_stats(X[i][c], X[j][c]); t.update(motif=m, rep=r, pair=f"{i}-{j}", N=N); rows.append(t)
    return rows
if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--nreps", type=int, default=100); ap.add_argument("--rep_offset", type=int, default=0, help="skip the first rep_offset common replicates (figure 4)"); ap.add_argument("--fig4_out", default="twin_size_scan_figure4.csv"); a = ap.parse_args()
    if not os.path.exists(O + "twin_size_scan_figure3_6000.csv"):
        with mp.Pool(a.jobs) as p: res = p.map(fig3_rep, range(100), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(O + "twin_size_scan_figure3_6000.csv", index=False); print("figure3 6000 done", flush=True)
    if not os.path.exists(O + a.fig4_out):
        F = files4(); common = sorted(set.intersection(*[{r for (mm, r) in F if mm == m} for m in MOT]))[a.rep_offset:a.rep_offset + a.nreps]; print("fig4 replicates:", common[:3], "...", common[-1], len(common), flush=True)
        with mp.Pool(a.jobs) as p: res = p.map(fig4_rep, [(m, r, F[(m, r)]) for r in common for m in MOT], chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(O + a.fig4_out, index=False); print("figure4 done", flush=True)
