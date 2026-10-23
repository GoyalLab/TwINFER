"""Flowchart statistics (step1_z, z_het, z_d, ...) + S/C/U/z at t1 and t2 for every gene pair of the network_sweep 5-edge networks (3 networks x 3 sims x 15 pairs = 135), clone-weighted infer.py steps
(run_6scenario_zscores.process_replicate on the two genes of the pair, 2,000 null draws), t1 = 1 h, t2 = 20 h. Pair class from the ground truth (classes()). usage: python e5_flowchart_pairs.py OUT_CSV [--jobs 12]"""
import glob, os, re, sys, argparse, numpy as np, pandas as pd, multiprocessing as mp
_argv = sys.argv; sys.argv = [sys.argv[0]]
from network_sweep_scenarios import classes
import scenario_flowchart_clone_weighted as SF
sys.argv = _argv
R = SF.R; R.N_SHUFFLES = 2000; _orig = R._load; R._load = lambda p: p if isinstance(p, pd.DataFrame) else _orig(p)
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; SIM = ROOT + "simulation_data/network_sweep/"; NET = "grn_n6_e5_pos50_density"
def one(a):
    f, net, sim, i, j, cls, x, y = a
    d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate", f"gene_{x+1}_mRNA", f"gene_{y+1}_mRNA"]).rename(columns={f"gene_{x+1}_mRNA": "gene_1_mRNA", f"gene_{y+1}_mRNA": "gene_2_mRNA"})
    if d[d.time_step.isin([1, 20])][["gene_1_mRNA", "gene_2_mRNA"]].std().min() == 0: return None
    rec = R.process_replicate(d, "e5", sim, n_cores=1); P = R._partition(d)
    for tag, key in (("t1", "t1_twins"), ("t2", "t2_twins")):
        for k, v in SF.scu(P[key]).items(): rec[f"{k}_{tag}"] = float(v)
        nt = rec[f"n_twins_{tag}"]; nsh = len(P["ac_left"]); nb = nt + nsh
        rec[f"z_rho_{tag}"] = float(rec[f"rho_{tag}"] / np.sqrt((nt * (0.5 + 0.5 * rec[f"h_x_{tag}"] * rec[f"h_y_{tag}"]) + nsh) / nb ** 2))
    rec = {k: v for k, v in rec.items() if not isinstance(v, (list, dict)) and not k.startswith(("gene_t", "twin_delta", "random_delta"))}
    rec.update(network=f"rep{net}", sim=sim, pair=f"{i+1}-{j+1}", cls=cls, x=x + 1, y=y + 1); return rec
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); tasks = []
    for net in (0, 1, 2):
        C = np.loadtxt(ROOT + f"input_data/network_sweep/{NET}_rep{net}.txt", delimiter=",").astype(int); cl = classes(C)
        for f in sorted(glob.glob(SIM + f"df_{NET}_rep{net}_rep*_*.csv")):
            sim = int(re.search(rf"{NET}_rep{net}_rep(\d+)_", f).group(1))
            tasks += [(f, net, sim, i, j, c, x, y) for (i, j), (c, x, y) in cl.items()]
    print(len(tasks), "pair tasks", flush=True)
    with mp.Pool(a.jobs) as p: res = [r for r in p.map(one, tasks, chunksize=1) if r]
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
