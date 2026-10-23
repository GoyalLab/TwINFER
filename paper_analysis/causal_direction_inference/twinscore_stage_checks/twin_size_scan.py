"""z_rho, z_C, z_U (twin-only designs) for N = 250, 500, 1000, 1500, 2000, 2500, 3000 twin pairs at 20 h, for the figure 2 two-gene scenarios and the figure 3 conditions (100 replicates each).
Nested subsets: one clone permutation per (scenario/condition, replicate) (RandomState(rep*100 + si + 12345)), N = first N clones, so every smaller design is contained in the larger ones; identical to the permutations used in twin_decomposition.py / figure3_decomposition.py.
Output: <O>/twin_size_scan_figure2.csv, <O>/twin_size_scan_figure3.csv   usage: python twin_size_scan.py [--jobs 8]"""
import argparse, glob, os, numpy as np, pandas as pd
from twin_decomposition import twin_stats
import plot_new_metrics_boxplots as P
import figure3_decomposition as F3
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; NS = [250, 500, 1000, 1500, 2000, 2500, 3000]
def rows_for(P2, cc, name_key, name, rep):
    out = []
    for N in NS:
        t = twin_stats(P2["gene_1"][cc[:N]], P2["gene_2"][cc[:N]]); t.update({name_key: name, "rep": rep, "N": N}); out.append(t)
    return out
def fig2_rep(rep):
    rows = []; jobs = [(name, cfgs, None, si) for si, (name, cfgs) in enumerate(P.SCEN)] + [(f"{n} (drift)", None, n, 7) for n in ("A_B", "A_to_B")]
    for name, cfgs, net, si in jobs:
        rng = np.random.RandomState(rep * 10 + si)
        if net: Pd = P.load20(glob.glob(f"{P.DRIFT}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")[0])
        elif len(cfgs) == 1: Pd = P.load20(P.find(cfgs[0], rep)[0])
        else:
            S_ = [P.load20(P.find(c_, rep)[0]) for c_ in cfgs]; sa, sb = rng.choice(P.NCL, 3000, replace=False), rng.choice(P.NCL, 3000, replace=False); Pd = {g: np.vstack([S_[0][g][sa], S_[1][g][sb]]) for g in P.G}
        rows += rows_for(Pd, np.random.RandomState(rep * 100 + si + 12345).permutation(P.NCL), "scenario", name, rep)
    return rows
def fig3_rep(rep):
    rows = []
    for si, cond in enumerate(F3.CONDS):
        s = F3.load(F3.find(cond, F3.REPS[cond][rep])); rows += rows_for(s[20], np.random.RandomState(rep * 100 + si + 12345).permutation(F3.NCL), "condition", cond, rep)
    return rows
if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    for fn, f, out in ((fig2_rep, "figure2", "twin_size_scan_figure2.csv"), (fig3_rep, "figure3", "twin_size_scan_figure3.csv")):
        if os.path.exists(O + out): continue
        with mp.Pool(a.jobs) as p: res = p.map(fn, range(100), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(O + out, index=False); print(f, "done", flush=True)
