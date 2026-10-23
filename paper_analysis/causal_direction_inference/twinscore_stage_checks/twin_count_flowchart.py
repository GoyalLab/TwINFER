"""Flowchart statistics (Stage I step1_z at t1, Stage II z_het at t1/t2, Stage III z_d) vs number of twins N for the 4 two-gene simulations (100 replicates each), clone-weighted infer.py steps via
run_6scenario_zscores helpers. Design = the Figure-2 / infer.py design: N random clones as t1 twins (1 h), N OTHER clones as t2 twins (20 h), plus 3,000 cross clones that give ONE cell to the t1 sample
(replicate 0) and the sister to the t2 sample (replicate 1) -> rho set at t1 = t1 twins + 3,000 cross cells, at t2 = t2 twins + their 3,000 sisters (as R._partition). Needs 2N + 3,000 <= 6,000 clones -> N <= 1,500.
Null draws: R.N_SHUFFLES (default 5,000, as the earlier flowchart table). One CSV per simulation file in OUT_DIR (resumable); aggregated to OUT_DIR/twin_count_flowchart.csv.
usage: python twin_count_flowchart.py OUT_DIR [--nreps 100] [--jobs 24] [--shuffles 5000] [--only scen:rep] [--ns 250,1500]"""
import os, sys, glob, argparse, numpy as np, pandas as pd, numba, multiprocessing as mp
from threadpoolctl import threadpool_limits
_argv = sys.argv; sys.argv = [sys.argv[0]]
import scenario_flowchart_clone_weighted as SF
sys.argv = _argv
R = SF.R; SEED = R.SEED; NS = (250, 376, 500, 712, 1000, 1500); NCROSS = 3000
from twinfer.inference.correlation_functions import assign_twin_id, identify_reg_if_multiple_states
KEEP = {"no_regulation": "A_B (single state)", "A_to_B": "A_to_B (single state)", "multistate_A_B": "A_B (two states)", "multistate_A_to_B": "A_to_B (two states)"}
OUT = None; NS_USE = NS
def one(a):
    scen, rep, path = a; dest = f"{OUT}/{scen}_rep_{rep}.csv"
    if os.path.exists(dest): return dest
    numba.set_num_threads(1); rows = []
    with threadpool_limits(limits=1):
        sim = R._load(path); clones = sim["clone_id"].drop_duplicates().to_numpy(); perm = np.random.default_rng(SEED + rep).permutation(clones); reps = sorted(sim["replicate"].unique())
        s1 = sim[sim.time_step == R.T1]; s2 = sim[sim.time_step == R.T2]
        for N in NS_USE:
            A, B, X = perm[:N], perm[N:2 * N], perm[2 * N:2 * N + NCROSS]
            t1r = s1[s1.clone_id.isin(A)].copy(); t2r = s2[s2.clone_id.isin(B)].copy()
            acl = s1[s1.clone_id.isin(X) & (s1.replicate == reps[0])].copy(); acr = s2[s2.clone_id.isin(X) & (s2.replicate == reps[1])].copy()
            rho1 = pd.concat([t1r, acl], ignore_index=True); rho2 = pd.concat([t2r, acr], ignore_index=True)
            t1 = assign_twin_id(t1r).reset_index(drop=True); t2 = assign_twin_id(t2r).reset_index(drop=True)
            st1, g1 = R._step1(rho1, SEED + 1)
            sa, tw1, rd1 = R._step2(rho1, t1, SEED + 271829, SEED + 271832, 1); sb, tw2, rd2 = R._step2(rho2, t2, SEED + 271830, SEED + 271833, 1)
            _, _, st3 = identify_reg_if_multiple_states(tw1, tw2, rd1, rd2, [R.PAIR], R.GENES, t1, t2, t1_raw=rho1, t2_raw=rho2, alpha=0.01, n_shuffles=R.N_SHUFFLES, shuffle_seed=SEED + 271831, unit="clone", n_cores_to_use=1)
            s3 = R._pv(st3) or {}
            rows.append(dict(sim=KEEP[scen], rep=rep, N=N, n_singletons=len(acl), step1_z_t1=st1["z"], step2_z_het_t1=sa["z_het"], step2_z_het_t2=sb["z_het"], step3_z_d=s3.get("z_d"), rho_t1=st1["rho"]))
    pd.DataFrame(rows).to_csv(dest + ".tmp", index=False); os.replace(dest + ".tmp", dest); return dest
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nreps", type=int, default=100); ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--shuffles", type=int, default=5000); ap.add_argument("--only", default=None); ap.add_argument("--ns", default=None); a = ap.parse_args()
    OUT = a.out; os.makedirs(OUT, exist_ok=True); R.N_SHUFFLES = a.shuffles; R.N_REPS = a.nreps
    if a.ns: NS_USE = tuple(int(x) for x in a.ns.split(","))
    tasks = [t for t in R.collect_tasks() if t[0] in KEEP]
    if a.only: tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in set(a.only.split(","))]
    print(len(tasks), "simulation files x N =", NS_USE, f"; shuffles {R.N_SHUFFLES}; {NCROSS} singleton (cross) cells", flush=True)
    with mp.Pool(a.jobs) as p:
        for i, d in enumerate(p.imap_unordered(one, tasks, chunksize=1)): print(f"[{i+1}/{len(tasks)}] {os.path.basename(d)}", flush=True)
    fs = sorted(glob.glob(f"{OUT}/*_rep_*.csv")); pd.concat([pd.read_csv(f) for f in fs]).to_csv(f"{OUT}/twin_count_flowchart.csv", index=False); print("aggregated", len(fs), "files")
