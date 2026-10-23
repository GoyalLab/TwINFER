"""Flowchart statistics (step1_z, z_het, z_d) + S/C/U/z at t1, t2 for the 3 pairs of every 3-gene motif replicate (Fan_out, Feed_forward, Mutual_regulation; figure_4 1k_per_motif_v2; first 100 common replicates),
clone-weighted infer.py steps via run_6scenario_zscores.process_replicate on the two genes of the pair (1,000 null draws), t1 = 1 h, t2 = 20 h. usage: python motif_flowchart_pairs.py OUT_CSV [--nreps 100] [--jobs 14]"""
import sys, argparse, numpy as np, pandas as pd, multiprocessing as mp
_argv = sys.argv; sys.argv = [sys.argv[0]]
import scenario_flowchart_clone_weighted as SF
from motif_replicates_zscores import files, MOT
from motif_shared_driver import PAIRS, TRUTH
sys.argv = _argv
R = SF.R; R.N_SHUFFLES = 1000; _orig = R._load; R._load = lambda p: p if isinstance(p, pd.DataFrame) else _orig(p)
def one(a):
    f, m, rep, (y, z) = a
    d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate", f"gene_{y}_mRNA", f"gene_{z}_mRNA"]).rename(columns={f"gene_{y}_mRNA": "gene_1_mRNA", f"gene_{z}_mRNA": "gene_2_mRNA"})
    if d[d.time_step.isin([1, 20])][["gene_1_mRNA", "gene_2_mRNA"]].std().min() == 0: return None
    rec = R.process_replicate(d, m, rep, n_cores=1); P = R._partition(d)
    for tag, key in (("t1", "t1_twins"), ("t2", "t2_twins")):
        for k, v in SF.scu(P[key]).items(): rec[f"{k}_{tag}"] = float(v)
        nt = rec[f"n_twins_{tag}"]; nsh = len(P["ac_left"]); nb = nt + nsh
        rec[f"z_rho_{tag}"] = float(rec[f"rho_{tag}"] / np.sqrt((nt * (0.5 + 0.5 * rec[f"h_x_{tag}"] * rec[f"h_y_{tag}"]) + nsh) / nb ** 2))
    rec = {k: v for k, v in rec.items() if not isinstance(v, (list, dict)) and not k.startswith(("gene_t", "twin_delta", "random_delta"))}
    rec.update(motif=m, rep=rep, pair=f"{y}-{z}", truth=TRUTH[m][(y, z)]); return rec
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nreps", type=int, default=100); ap.add_argument("--jobs", type=int, default=14); a = ap.parse_args()
    Fl = files(); common = sorted(set.intersection(*[{r for (mm, r) in Fl if mm == m} for m in MOT]))[:a.nreps]
    tasks = [(Fl[(m, r)], m, r, p) for m in MOT for r in common for p in PAIRS]; print(len(tasks), "tasks", flush=True)
    with mp.Pool(a.jobs) as p: res = [x for x in p.imap_unordered(one, tasks, chunksize=1) if x]
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
