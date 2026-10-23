"""Part 2 of scale_heterogeneity_test.py: the four z_U estimators on the real simulations (baseline design: 1,500 twin pairs + 3,000 singletons, 20 h, mRNA, 100 replicates).
usage: python scale_heterogeneity_real.py OUT_CSV [--jobs 12]"""
import sys, glob, argparse, numpy as np, pandas as pd, multiprocessing as mp
_argv = sys.argv; sys.argv = [sys.argv[0]]
import scale_heterogeneity_test as ST
sys.argv = _argv
RS = ST.RS; R = RS.R; SEED = R.SEED; N = 1500; NS_ = 3000
def build(sim, rep, name):
    clones = sim["clone_id"].drop_duplicates().to_numpy(); perm = np.random.default_rng(SEED + rep).permutation(clones); reps = sorted(sim["replicate"].unique()); s2 = sim[sim.time_step == R.T2]
    tw = s2[s2.clone_id.isin(perm[N:2 * N])]; sg = s2[s2.clone_id.isin(perm[2 * N:2 * N + NS_]) & (s2.replicate == reps[1])]
    M = {g: tw.pivot(index="clone_id", columns="replicate", values=f"{g}_mRNA").values.astype(float) for g in ("gene_1", "gene_2")}
    z = ST.estimators(M["gene_1"], M["gene_2"], sg["gene_1_mRNA"].to_numpy(float), sg["gene_2_mRNA"].to_numpy(float)); return dict(sim=name, rep=rep, **z)
def sims(a): scen, rep, path = a; return build(R._load(path), rep, RS.KEEP[scen])
def drift(a): net, rep = a; fs = glob.glob(f"{RS.DR}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv"); return build(pd.read_csv(fs[0]), rep, f"{net} (drift)")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); R.N_REPS = 100
    tasks = [t for t in R.collect_tasks() if t[0] in RS.KEEP]
    with mp.Pool(a.jobs) as p: res = p.map(sims, tasks, chunksize=1) + p.map(drift, [(n, r) for n in ("A_B", "A_to_B") for r in range(100)])
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
