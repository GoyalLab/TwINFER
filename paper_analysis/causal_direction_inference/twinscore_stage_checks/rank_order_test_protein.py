"""Same comparison as rank_order_test.py (rank-first Pearson on differences vs infer.py difference-first Spearman) but on PROTEIN counts of the same simulations (few ties), N = 1,500 and 3,000, 100 replicates.
If the rank-first advantage seen for mRNA is due to ties in the raw differences it should disappear here. usage: python rank_order_test_protein.py OUT_CSV [--jobs 12]"""
import sys, argparse, numpy as np, pandas as pd, multiprocessing as mp
from scipy.stats import rankdata
_argv = sys.argv; sys.argv = [sys.argv[0]]
import rank_order_test as RO
sys.argv = _argv
R = RO.R; SEED = RO.SEED; KEEP = RO.KEEP
def sims(a):
    scen, rep, path = a; sim = R._load(path); clones = sim["clone_id"].drop_duplicates().to_numpy(); perm = np.random.default_rng(SEED + rep).permutation(clones); s2 = sim[sim.time_step == R.T2]; out = []
    for N in (1500, 3000):
        tw = s2[s2.clone_id.isin(perm[N:2 * N])]; M = {g: tw.pivot(index="clone_id", columns="replicate", values=f"{g}_protein").values.astype(float) for g in ("gene_1", "gene_2")}
        S, C, zc = RO.zC(M["gene_1"], M["gene_2"]); a_, b_, tx, ty = RO.rho_delta(M["gene_1"], M["gene_2"])
        out.append(dict(sim=KEEP[scen], rep=rep, N=N, z_C=zc, z_U_rank_first=a_ * np.sqrt(N), z_U_diff_first=b_ * np.sqrt(N), rho_D_rank_first=a_, rho_D_diff_first=b_, tie_x=tx, tie_y=ty))
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); R.N_REPS = 100
    tasks = [t for t in R.collect_tasks() if t[0] in KEEP]
    with mp.Pool(a.jobs) as p: res = [r for rr in p.map(sims, tasks, chunksize=1) for r in rr]
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
