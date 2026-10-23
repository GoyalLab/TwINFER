"""Design X on the same clone permutations as the twin-only designs (x_design_common.x_stats) for (a) all network simulations (permutation = crc32(raw|sim), as twin_decomposition.net_file) and (b) the two-gene scenarios
(permutation = RandomState(rep*100+si+12345), as twin_decomposition.twogene_rep).  Caches <O>/twin_X_common_networks.csv, <O>/twin_X_common_twogene.csv.   usage: python x_design_networks_twogene.py [--jobs 6]"""
import argparse, glob, os, re, zlib, numpy as np, pandas as pd
from x_design_common import x_stats
import twin_design_zscores as T
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
def net_file(a):
    raw, sim, f = a; cols = pd.read_csv(f, nrows=0).columns; genes = sorted(int(re.match(r"gene_(\d+)_mRNA", c).group(1)) for c in cols if re.match(r"gene_\d+_mRNA$", c))
    d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate"] + [f"gene_{g}_mRNA" for g in genes]); d = d[d.time_step == 20].sort_values(["clone_id", "replicate"]); ncl = d.clone_id.nunique()
    X = {g: d[f"gene_{g}_mRNA"].to_numpy(float).reshape(ncl, 2) for g in genes}; q = np.random.RandomState(zlib.crc32(f"{raw}|{sim}".encode()) % 2**31).permutation(ncl)
    pick = np.random.RandomState(zlib.crc32(f"{raw}|{sim}|pick".encode()) % 2**31).randint(0, 2, 3000); rows = []
    for i in genes:
        for j in genes:
            if j <= i or X[i].std() == 0 or X[j].std() == 0: continue
            o = x_stats(X[i], X[j], q, pick); o.update(raw=raw, sim=sim, x=i, y=j); rows.append(o)
    return rows
def twogene_rep(rep):
    import plot_new_metrics_boxplots as P
    rows = []; jobs = [(name, cfgs, None, si) for si, (name, cfgs) in enumerate(P.SCEN)] + [(f"{n} (drift)", None, n, 7) for n in ("A_B", "A_to_B")]
    for name, cfgs, net, si in jobs:
        rng = np.random.RandomState(rep * 10 + si)
        if net: Pd = P.load20(glob.glob(f"{P.DRIFT}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")[0])
        elif len(cfgs) == 1: Pd = P.load20(P.find(cfgs[0], rep)[0])
        else:
            S_ = [P.load20(P.find(c_, rep)[0]) for c_ in cfgs]; sa, sb = rng.choice(P.NCL, 3000, replace=False), rng.choice(P.NCL, 3000, replace=False); Pd = {g: np.vstack([S_[0][g][sa], S_[1][g][sb]]) for g in P.G}
        q = np.random.RandomState(rep * 100 + si + 12345).permutation(P.NCL); o = x_stats(Pd["gene_1"], Pd["gene_2"], q, np.random.RandomState(rep * 100 + si + 54321).randint(0, 2, 3000)); o.update(scenario=name, rep=rep); rows.append(o)
    return rows
if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=6); a = ap.parse_args()
    if not os.path.exists(O + "twin_X_common_twogene.csv"):
        with mp.Pool(a.jobs) as p: res = p.map(twogene_rep, range(100), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(O + "twin_X_common_twogene.csv", index=False); print("twogene done", flush=True)
    if not os.path.exists(O + "twin_X_common_networks.csv"):
        with mp.Pool(a.jobs) as p: res = p.map(net_file, T.files(), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(O + "twin_X_common_networks.csv", index=False); print("networks done", flush=True)
