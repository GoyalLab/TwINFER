"""Spearman (ranks, as everywhere so far) vs Pearson (raw mRNA counts) vs Pearson on log1p counts in z_rho, z_C, z_U, on the 4 two-gene simulations (figure_2_simulations_1000) and the 2 drift simulations (drift_lowhigh_Kramp), 100 replicates, 20 h.
Designs with 6,000 cells: A = 3,000 twin pairs (twin-only); B = 1,500 twin pairs + 3,000 singletons (the 4 simulations only; same clones as twin_count_flowchart / twin_count_power_singletons). Same formulas for both
(sigma_C^2 = (1+S^2+h_x h_y)/(2N); z_U = rho_hat_Delta*sqrt(N); sigma_rho^2 = (N(1/2+1/2 h_x h_y)+n_single)/(N+n_single)^2). usage: python pearson_vs_spearman.py OUT_CSV [--jobs 12]"""
import sys, glob, argparse, numpy as np, pandas as pd, multiprocessing as mp
_argv = sys.argv; sys.argv = [sys.argv[0]]
import scenario_flowchart_clone_weighted as SF
sys.argv = _argv
import figure2_1000_zscores as F
from u_time_pool import zU
from motif_replicates_zscores import zC, zRho
R = SF.R; SEED = R.SEED
KEEP = {"no_regulation": "A_B (single state)", "A_to_B": "A_to_B (single state)", "multistate_A_B": "A_B (two states)", "multistate_A_to_B": "A_to_B (two states)"}
METHODS = None   # set below (after pearson/spearman are defined)
DR = "/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/"
def pearson(X, Y, xs=None, ys=None):
    n = len(X); rx = X - X.mean(); ry = Y - Y.mean(); Mx = ((rx ** 2).sum(1) / 2).mean(); My = ((ry ** 2).sum(1) / 2).mean(); nrm = np.sqrt(Mx * My)
    S = ((rx[:, 0] * ry[:, 0] + rx[:, 1] * ry[:, 1]) / 2).mean() / nrm; C = ((rx[:, 0] * ry[:, 1] + rx[:, 1] * ry[:, 0]) / 2).mean() / nrm
    hx = max(0., (rx[:, 0] * rx[:, 1]).mean() / Mx); hy = max(0., (ry[:, 0] * ry[:, 1]).mean() / My)
    zc = np.sign(S) * C / np.sqrt((1 + S ** 2 + hx * hy) / (2 * n)); dx = rx[:, 0] - rx[:, 1]; dy = ry[:, 0] - ry[:, 1]; zu = (dx * dy).sum() / np.sqrt((dx ** 2).sum() * (dy ** 2).sum()) * np.sqrt(n)
    ns = 0 if xs is None else len(xs); vx = X.ravel() if ns == 0 else np.concatenate([X.ravel(), xs]); vy = Y.ravel() if ns == 0 else np.concatenate([Y.ravel(), ys])
    w = np.concatenate([np.full(2 * n, 0.5), np.ones(ns)]); rho = F.wcorr(vx, vy, w); z_rho = rho / np.sqrt((n * (0.5 + 0.5 * hx * hy) + ns) / (n + ns) ** 2)
    return dict(z_rho=z_rho, z_C=zc, z_U=zu)
def spearman(X, Y, xs=None, ys=None):
    n = len(X); S, C, zc = zC(X, Y); zu = zU(X, Y)
    if xs is None: z_rho = zRho(X, Y)
    else:
        ns = len(xs); w = np.concatenate([np.full(2 * n, 0.5), np.ones(ns)]); rk = {}; h = {}
        for g, (T, s1) in {"x": (X, xs), "y": (Y, ys)}.items(): rk[g] = F.wrank(np.concatenate([T.ravel(), s1]), w); h[g] = F.h_of(rk[g][:2 * n].reshape(n, 2))
        z_rho = F.wcorr(rk["x"], rk["y"], w) / np.sqrt((n * (0.5 + 0.5 * h["x"] * h["y"]) + ns) / (n + ns) ** 2)
    return dict(z_rho=z_rho, z_C=zc, z_U=zu)
METHODS = (("Spearman", lambda *a: spearman(*a)), ("Pearson", lambda *a: pearson(*a)), ("Pearson log1p", lambda *a: pearson(*[np.log1p(x) for x in a])))
def sims(a):
    scen, rep, path = a; sim = R._load(path); clones = sim["clone_id"].drop_duplicates().to_numpy(); perm = np.random.default_rng(SEED + rep).permutation(clones); reps = sorted(sim["replicate"].unique()); s2 = sim[sim.time_step == R.T2]; rows = []
    for design, N in (("A: 3,000 twin pairs", 3000), ("B: 1,500 pairs + 3,000 singletons", 1500)):
        B = perm[N:2 * N]; tw = s2[s2.clone_id.isin(B)]; M = {g: tw.pivot(index="clone_id", columns="replicate", values=f"{g}_mRNA").values for g in ("gene_1", "gene_2")}; sg = None
        if N == 1500: sg = s2[s2.clone_id.isin(perm[2 * N:2 * N + 3000]) & (s2.replicate == reps[1])]; sg = (sg["gene_1_mRNA"].to_numpy(float), sg["gene_2_mRNA"].to_numpy(float))
        args = (M["gene_1"].astype(float), M["gene_2"].astype(float)) + (sg if sg else ())
        for nm, f in METHODS: rows.append(dict(sim=KEEP[scen], rep=rep, design=design, method=nm, **f(*args)))
    return rows
def drift(a):
    net, rep = a; fs = glob.glob(f"{DR}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv"); P = F.load(fs[0])[20]; perm = np.random.RandomState(rep * 10 + 7).permutation(P["gene_1"].shape[0]); c = perm[:3000]; rows = []
    args = (P["gene_1"][c].astype(float), P["gene_2"][c].astype(float))
    for nm, f in METHODS: rows.append(dict(sim=f"{net} (drift)", rep=rep, design="A: 3,000 twin pairs", method=nm, **f(*args)))
    return rows
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); R.N_REPS = 100
    tasks = [t for t in R.collect_tasks() if t[0] in KEEP]
    with mp.Pool(a.jobs) as p: res = [r for rr in p.map(sims, tasks, chunksize=1) for r in rr] + [r for rr in p.map(drift, [(n, r) for n in ("A_B", "A_to_B") for r in range(100)]) for r in rr]
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
