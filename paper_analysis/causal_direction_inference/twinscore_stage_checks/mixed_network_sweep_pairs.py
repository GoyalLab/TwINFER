"""S / C / U / z-scores for EVERY gene pair of every mixed_network_sweep simulation (48 networks x 3 sim replicates = 144 files), Figure-2 design (B = 20 h: 1,500 twins + 3,000 singletons; A = 1 h), mRNA only.
Pair class from the ground-truth topology (input_data/mixed_network_sweep/<net>.txt, C[i,j]!=0 => gene i+1 regulates j+1): direct edge | indirect | co-regulated | unrelated (classes() of network_sweep_scenarios).
Output: analysis_data/mixed_network_sweep/twinscore_pairs_SCU.csv  usage: python mixed_network_sweep_pairs.py [--jobs 12]"""
import glob, os, re, argparse, numpy as np, pandas as pd, multiprocessing as mp, sys
sys.argv_backup = sys.argv; sys.argv = [sys.argv[0]]          # network_sweep_scenarios parses argv at import
import figure2_1000_zscores as F
from network_sweep_scenarios import classes
sys.argv = sys.argv_backup
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; SIM = ROOT + "simulation_data/mixed_network_sweep/"; OUT = ROOT + "analysis_data/mixed_network_sweep/twinscore_pairs_SCU.csv"
def one(f):
    m = re.search(r"df_(grn_.+?_rep\d+)_rep(\d+)_\d{8}_", os.path.basename(f)); net, sim = m.group(1), int(m.group(2))
    C = np.loadtxt(ROOT + f"input_data/mixed_network_sweep/{net}.txt", delimiter=",").astype(int); n = C.shape[0]; cl = classes(C)
    d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate"] + [f"gene_{k}_mRNA" for k in range(1, n + 1)])
    P = {t: {k: d[d.time_step == t].pivot(index="clone_id", columns="replicate", values=f"gene_{k}_mRNA").values for k in range(1, n + 1)} for t in (1, 20)}
    rows = []
    for (i, j), (cls, x, y) in cl.items():
        gx, gy = x + 1, y + 1; rng = np.random.RandomState(7 * sim + 100 * i + j)
        if P[20][gx].std() == 0 or P[20][gy].std() == 0: continue
        o = F.analyse({"gene_1": P[1][gx], "gene_2": P[1][gy]}, {"gene_1": P[20][gx], "gene_2": P[20][gy]}, rng, 200)
        o.update(net=net, sim=sim, n_genes=n, pair=f"{i+1}-{j+1}", cls=cls, x=gx, y=gy, mutual=int(C[x, y] != 0 and C[y, x] != 0)); rows.append(o)
    return rows
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args()
    files = [f for f in sorted(glob.glob(SIM + "df_grn_*.csv"))]; print(len(files), "files", flush=True)
    with mp.Pool(a.jobs) as p: res = p.map(one, files, chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(OUT, index=False); print("done ->", OUT)
