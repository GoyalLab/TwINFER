"""Analytic power vs number of twins in the simulations. For N in 250..6000 random twin clones (twin-only design, sample B = 20 h, no singletons) compute z_rho, z_C, z_U (analytic, helpers of
motif_replicates_zscores / u_time_pool) for (a) the 4 two-gene simulations (A_B, A_to_B, A_B two states, A_to_B two states; figure_2_simulations_1000, 100 replicates each) and (b) the 9 gene pairs of the 3-gene
motifs (figure_4 1k_per_motif_v2, 100 replicates). usage: python twin_count_power.py OUT_DIR [--nreps 100] [--jobs 12]"""
import sys, os, argparse, numpy as np, pandas as pd, multiprocessing as mp
import figure2_1000_zscores as F
from u_time_pool import zU
from motif_replicates_zscores import files, load as mload, zC, zRho, MOT
from motif_shared_driver import PAIRS, TRUTH
NS = (250, 376, 500, 712, 1000, 1500, 3000, 6000)      # 376 = FM01 clones with exactly 2 cells, 712 = FM01 clones with >=2 cells
def stats(X, Y): S, C, zc = zC(X, Y); return dict(S=S, C=C, z_C=zc, z_U=zU(X, Y), z_rho=zRho(X, Y))
def two_gene(rep):
    rows = []
    for si, (name, cfgs) in enumerate(F.SCEN):
        rng = np.random.RandomState(rep * 10 + si); srcs = []
        for c in cfgs:
            p, n = F.find(c, rep); srcs.append(F.load(p))
        if len(srcs) == 1: P = srcs[0][F.T2]
        else:
            sa, sb = rng.choice(F.NCL, 3000, replace=False), rng.choice(F.NCL, 3000, replace=False)
            P = {g: np.vstack([srcs[0][F.T2][g][sa], srcs[1][F.T2][g][sb]]) for g in F.G}
        perm = rng.permutation(F.NCL)
        for N in NS:
            c = perm[:N]; o = stats(P["gene_1"][c], P["gene_2"][c]); o.update(kind="2gene", sim=name, rep=rep, N=N); rows.append(o)
    return rows
def motif(a):
    m, r, f = a; D = mload(f); rng = np.random.RandomState(r); perm = rng.permutation(D[20][1].shape[0]); rows = []
    for N in NS:
        c = perm[:N]
        for (i, j) in PAIRS:
            o = stats(D[20][i][c], D[20][j][c]); o.update(kind="motif", sim=m, pair=f"{i}-{j}", truth=TRUTH[m][(i, j)], rep=r, N=N); rows.append(o)
    return rows
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nreps", type=int, default=100); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    Fl = files(); common = sorted(set.intersection(*[{r for (mm, r) in Fl if mm == m} for m in MOT]))[:a.nreps]
    with mp.Pool(a.jobs) as p:
        r2 = [x for rr in p.map(two_gene, range(a.nreps)) for x in rr]; rm = [x for rr in p.map(motif, [(m, r, Fl[(m, r)]) for m in MOT for r in common]) for x in rr]
    pd.DataFrame(r2).to_csv(f"{a.out}/twin_count_power_2gene.csv", index=False); pd.DataFrame(rm).to_csv(f"{a.out}/twin_count_power_motifs.csv", index=False); print("done", len(r2), len(rm))
