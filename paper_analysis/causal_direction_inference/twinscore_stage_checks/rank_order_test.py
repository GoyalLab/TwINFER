"""rho_hat_Delta computed two ways on the same twin pairs (20 h, twin-only, N twin pairs):
  rank-first (PDF / analytic pipeline):   rank the counts over all 2N twin cells, take within-pair differences of the ranks, Pearson correlation of the differences;
  difference-first (infer.py rho_Delta):  take within-pair differences of raw counts, rank those N differences, Pearson of the ranks (= Spearman of the differences).
z_U = rho_hat_Delta * sqrt(N) for both; z_C (rank-first) is the same in both so the PDF rule (z_C, z_U) can be applied with each. 4 two-gene simulations + 2 drift, 100 replicates; N = 376, 712, 1500, 3000.
usage: python rank_order_test.py OUT_CSV [--jobs 12]"""
import sys, glob, argparse, numpy as np, pandas as pd, multiprocessing as mp
from scipy.stats import rankdata
_argv = sys.argv; sys.argv = [sys.argv[0]]
import scenario_flowchart_clone_weighted as SF
sys.argv = _argv
import figure2_1000_zscores as F
from motif_replicates_zscores import zC
R = SF.R; SEED = R.SEED; NS = (376, 712, 1500, 3000)
KEEP = {"no_regulation": "A_B (single state)", "A_to_B": "A_to_B (single state)", "multistate_A_B": "A_B (two states)", "multistate_A_to_B": "A_to_B (two states)"}
DR = "/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/"
def pear(a, b): a = a - a.mean(); b = b - b.mean(); return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
def rho_delta(X, Y):
    n = len(X); rx = rankdata(X.ravel()).reshape(n, 2); ry = rankdata(Y.ravel()).reshape(n, 2)
    rank_first = pear(rx[:, 0] - rx[:, 1], ry[:, 0] - ry[:, 1])
    diff_first = pear(rankdata(X[:, 0] - X[:, 1]), rankdata(Y[:, 0] - Y[:, 1]))
    return rank_first, diff_first, float(np.mean(X[:, 0] == X[:, 1])), float(np.mean(Y[:, 0] == Y[:, 1]))
def rows_for(sim, rep, X, Y, N):
    S, C, zc = zC(X, Y); a, b, tx, ty = rho_delta(X, Y)
    return dict(sim=sim, rep=rep, N=N, z_C=zc, z_U_rank_first=a * np.sqrt(N), z_U_diff_first=b * np.sqrt(N), rho_D_rank_first=a, rho_D_diff_first=b, tie_x=tx, tie_y=ty)
def sims(a):
    scen, rep, path = a; sim = R._load(path); clones = sim["clone_id"].drop_duplicates().to_numpy(); perm = np.random.default_rng(SEED + rep).permutation(clones); s2 = sim[sim.time_step == R.T2]; out = []
    for N in NS:
        tw = s2[s2.clone_id.isin(perm[N:2 * N])]; M = {g: tw.pivot(index="clone_id", columns="replicate", values=f"{g}_mRNA").values.astype(float) for g in ("gene_1", "gene_2")}
        out.append(rows_for(KEEP[scen], rep, M["gene_1"], M["gene_2"], N))
    return out
def drift(a):
    net, rep = a; fs = glob.glob(f"{DR}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv"); P = F.load(fs[0])[20]; perm = np.random.RandomState(rep * 10 + 7).permutation(P["gene_1"].shape[0]); out = []
    for N in NS: c = perm[:N]; out.append(rows_for(f"{net} (drift)", rep, P["gene_1"][c].astype(float), P["gene_2"][c].astype(float), N))
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); R.N_REPS = 100
    tasks = [t for t in R.collect_tasks() if t[0] in KEEP]
    with mp.Pool(a.jobs) as p: res = [r for rr in p.map(sims, tasks, chunksize=1) for r in rr] + [r for rr in p.map(drift, [(n, r) for n in ("A_B", "A_to_B") for r in range(100)]) for r in rr]
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
