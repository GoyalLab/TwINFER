"""Rank-first vs difference-first rho_hat_Delta in the BASELINE design: 1,500 twin pairs + 3,000 singleton cells (sisters of 3,000 cross clones, replicate 1), 20 h; 4 two-gene simulations + 2 drift, 100 replicates.
rank-first (PDF / analytic): weighted midranks over ALL 6,000 cells (twin cells w=1/2, singletons w=1), twin differences of those ranks, Pearson of the differences.
difference-first (infer.py): raw count differences of the 1,500 twin pairs, ranked, Pearson of the ranks. z_U = rho_hat_Delta*sqrt(1500). z_C from the twin part of the rank-first ranks (as figure2_1000_zscores.analyse).
counts = mRNA or protein columns. usage: python rank_order_singletons.py OUT_CSV mRNA|protein [--jobs 12]"""
import sys, glob, argparse, numpy as np, pandas as pd, multiprocessing as mp
from scipy.stats import rankdata
_argv = sys.argv; sys.argv = [sys.argv[0]]
import scenario_flowchart_clone_weighted as SF
sys.argv = _argv
import figure2_1000_zscores as F
R = SF.R; SEED = R.SEED; N = 1500; NS_ = 3000; KIND = "mRNA"
KEEP = {"no_regulation": "A_B (single state)", "A_to_B": "A_to_B (single state)", "multistate_A_B": "A_B (two states)", "multistate_A_to_B": "A_to_B (two states)"}
DR = "/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/"
def pear(a, b): a = a - a.mean(); b = b - b.mean(); return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
def stats(Xt, Yt, xs, ys):
    n = len(Xt); w = np.concatenate([np.full(2 * n, 0.5), np.ones(len(xs))])
    rx = F.wrank(np.concatenate([Xt.ravel(), xs]), w); ry = F.wrank(np.concatenate([Yt.ravel(), ys]), w); a = rx[:2 * n].reshape(n, 2); b = ry[:2 * n].reshape(n, 2)
    rank_first = pear(a[:, 0] - a[:, 1], b[:, 0] - b[:, 1]); diff_first = pear(rankdata(Xt[:, 0] - Xt[:, 1]), rankdata(Yt[:, 0] - Yt[:, 1]))
    ax = a - a.mean(); by = b - b.mean(); Mx = ((ax ** 2).sum(1) / 2).mean(); My = ((by ** 2).sum(1) / 2).mean(); nrm = np.sqrt(Mx * My)
    S = ((ax[:, 0] * by[:, 0] + ax[:, 1] * by[:, 1]) / 2).mean() / nrm; C = ((ax[:, 0] * by[:, 1] + ax[:, 1] * by[:, 0]) / 2).mean() / nrm
    hx = max(0., (ax[:, 0] * ax[:, 1]).mean() / Mx); hy = max(0., (by[:, 0] * by[:, 1]).mean() / My); zc = np.sign(S) * C / np.sqrt((1 + S ** 2 + hx * hy) / (2 * n))
    return dict(z_C=zc, z_U_rank_first=rank_first * np.sqrt(n), z_U_diff_first=diff_first * np.sqrt(n), rho_D_rank_first=rank_first, rho_D_diff_first=diff_first)
def build(sim, rep, name):
    clones = sim["clone_id"].drop_duplicates().to_numpy(); perm = np.random.default_rng(SEED + rep).permutation(clones); reps = sorted(sim["replicate"].unique()); s2 = sim[sim.time_step == R.T2]
    tw = s2[s2.clone_id.isin(perm[N:2 * N])]; sg = s2[s2.clone_id.isin(perm[2 * N:2 * N + NS_]) & (s2.replicate == reps[1])]
    M = {g: tw.pivot(index="clone_id", columns="replicate", values=f"{g}_{KIND}").values.astype(float) for g in ("gene_1", "gene_2")}
    o = stats(M["gene_1"], M["gene_2"], sg["gene_1_" + KIND].to_numpy(float), sg["gene_2_" + KIND].to_numpy(float)); o.update(sim=name, rep=rep, n_single=len(sg), tie_x=float(np.mean(M["gene_1"][:, 0] == M["gene_1"][:, 1]))); return o
def sims(a): scen, rep, path = a; return build(R._load(path), rep, KEEP[scen])
def drift(a): net, rep = a; fs = glob.glob(f"{DR}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv"); return build(pd.read_csv(fs[0]), rep, f"{net} (drift)")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("kind"); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); KIND = a.kind; R.N_REPS = 100
    tasks = [t for t in R.collect_tasks() if t[0] in KEEP]
    with mp.Pool(a.jobs) as p: res = p.map(sims, tasks, chunksize=1) + p.map(drift, [(n, r) for n in ("A_B", "A_to_B") for r in range(100)])
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
