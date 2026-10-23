"""Analytic z_rho / z_C / z_U vs number of twins N with the SAME design as twin_count_flowchart.py: N random twin clones (B sample, 20 h) + 3,000 singleton cells (sisters of the 3,000 cross clones, replicate 1).
z_rho = weighted Pearson of midranks over the 2N twin cells (weight 1/2) + 3,000 singletons (weight 1), sigma^2 = (N(1/2+1/2 h_x h_y)+3000)/(N+3000)^2 (Figure-2 design formula, h floored at 0);
z_C and z_U from the N twin pairs only (as before). Same clone draw (seed, permutation) as the flowchart run. usage: python twin_count_power_singletons.py OUT_CSV [--nreps 100] [--jobs 12]"""
import sys, os, argparse, numpy as np, pandas as pd, multiprocessing as mp
_argv = sys.argv; sys.argv = [sys.argv[0]]
import scenario_flowchart_clone_weighted as SF
sys.argv = _argv
import figure2_1000_zscores as F
from u_time_pool import zU
from motif_replicates_zscores import zC
R = SF.R; SEED = R.SEED; NS = (250, 376, 500, 712, 1000, 1500); NCROSS = 3000
KEEP = {"no_regulation": "A_B (single state)", "A_to_B": "A_to_B (single state)", "multistate_A_B": "A_B (two states)", "multistate_A_to_B": "A_to_B (two states)"}
def one(a):
    scen, rep, path = a; sim = R._load(path); clones = sim["clone_id"].drop_duplicates().to_numpy(); perm = np.random.default_rng(SEED + rep).permutation(clones); reps = sorted(sim["replicate"].unique())
    s2 = sim[sim.time_step == R.T2]; rows = []
    for N in NS:
        B, X = perm[N:2 * N], perm[2 * N:2 * N + NCROSS]
        tw = s2[s2.clone_id.isin(B)]; sg = s2[s2.clone_id.isin(X) & (s2.replicate == reps[1])]
        mats = {g: tw.pivot(index="clone_id", columns="replicate", values=f"{g}_mRNA").values for g in ("gene_1", "gene_2")}; ns = len(sg)
        w = np.concatenate([np.full(2 * N, 0.5), np.ones(ns)]); rk, h = {}, {}
        for g in mats:
            v = np.concatenate([mats[g].ravel(), sg[f"{g}_mRNA"].to_numpy()]); rk[g] = F.wrank(v, w); h[g] = F.h_of(rk[g][:2 * N].reshape(N, 2))
        rho = F.wcorr(rk["gene_1"], rk["gene_2"], w); sig = np.sqrt((N * (0.5 + 0.5 * h["gene_1"] * h["gene_2"]) + ns) / (N + ns) ** 2)
        Sv, Cv, zc = zC(mats["gene_1"], mats["gene_2"]); rows.append(dict(sim=KEEP[scen], rep=rep, N=N, n_singletons=ns, z_rho_sing=rho / sig, z_C=zc, z_U=zU(mats["gene_1"], mats["gene_2"]), S=Sv, C=Cv))
    return rows
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nreps", type=int, default=100); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args()
    R.N_REPS = a.nreps; tasks = [t for t in R.collect_tasks() if t[0] in KEEP]; print(len(tasks), "files", flush=True)
    with mp.Pool(a.jobs) as p: res = [r for rr in p.map(one, tasks, chunksize=1) for r in rr]
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
