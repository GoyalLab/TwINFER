"""Analytic z_rho / z_C / z_U vs number of twins N on the drift simulations (drift_lowhigh_Kramp, A_B and A_to_B, 100 replicates; sample = 20 h, twin-only design, no singletons), same helpers as twin_count_power.py.
usage: python twin_count_power_drift.py OUT_CSV [--nreps 100] [--jobs 12]"""
import glob, argparse, numpy as np, pandas as pd, multiprocessing as mp
import figure2_1000_zscores as F
from u_time_pool import zU
from motif_replicates_zscores import zC, zRho
ROOT = "/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/"; NS = (250, 376, 500, 712, 1000, 1500, 3000, 6000)
def one(a):
    net, rep = a; fs = glob.glob(f"{ROOT}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv"); assert len(fs) == 1, (net, rep, len(fs))
    P = F.load(fs[0])[20]; perm = np.random.RandomState(rep * 10 + 7).permutation(P["gene_1"].shape[0]); rows = []
    for N in NS:
        c = perm[:N]; X, Y = P["gene_1"][c], P["gene_2"][c]; S, C, zc = zC(X, Y)
        rows.append(dict(sim=f"{net} (drift)", rep=rep, N=N, z_rho=zRho(X, Y), z_C=zc, z_U=zU(X, Y), S=S, C=C))
    return rows
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nreps", type=int, default=100); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args()
    with mp.Pool(a.jobs) as p: res = [r for rr in p.map(one, [(n, r) for n in ("A_B", "A_to_B") for r in range(a.nreps)]) for r in rr]
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
