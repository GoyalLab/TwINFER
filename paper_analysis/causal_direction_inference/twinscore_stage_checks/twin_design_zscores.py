"""z_rho / z_C / z_U for every gene pair of every simulation under twin-only designs at t = 20 h: N twin clones (both sisters), no singletons.  N = 1500 (3,000 cells), N = 3000 (6,000 cells, the same cell count as 1,500 twins + 3,000 singletons) and N = 6000 (12,000 cells, all clones).
Clones are a seeded random subset per (network, sim).  Uses zRho (twin-only Stage I), zC, zU exactly as network_sweep_twin_only.py / motif_replicates_zscores.py.
Output: analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/twin_design_zscores.csv   usage: python twin_design_zscores.py [--jobs 8] [--limit K]"""
import glob, os, re, argparse, zlib, numpy as np, pandas as pd, multiprocessing as mp
from u_time_pool import zU
from motif_replicates_zscores import zC, zRho
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; OUT = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/twin_design_zscores.csv"
PATS = [(A + "simulation_data/mixed_network_sweep/df_grn_*.csv", r"df_(.+?_rep\d+)_rep(\d+)_\d{8}_"), (A + "simulation_data/network_sweep_final/df_grn_*.csv", r"df_(.+?_rep\d+)_rep(\d+)_\d{8}_"),
        (A + "simulation_data/network_sweep_final/e13_pos100/df_*.csv", r"_ncells_6000_(n6_e13_pos100_rep\d+)_rep_(\d+)_")]
def files():
    best = {}
    for g, rx in PATS:
        for f in glob.glob(g):
            if os.path.basename(f).startswith("simulation_before_division_df_"): continue
            m = re.search(rx, os.path.basename(f))
            if not m: continue
            k = (m.group(1), int(m.group(2)))
            if k not in best or os.path.getmtime(f) > os.path.getmtime(best[k]): best[k] = f           # stale duplicates: keep the latest file
    return sorted((k[0], k[1], f) for k, f in best.items())
def one(a):
    raw, sim, f = a; cols = pd.read_csv(f, nrows=0).columns; genes = sorted(int(re.match(r"gene_(\d+)_mRNA", c).group(1)) for c in cols if re.match(r"gene_\d+_mRNA$", c))
    d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate"] + [f"gene_{g}_mRNA" for g in genes]); d = d[d.time_step == 20].sort_values(["clone_id", "replicate"])
    ncl = d.clone_id.nunique(); assert len(d) == 2 * ncl, (f, len(d), ncl)
    X = {g: d[f"gene_{g}_mRNA"].to_numpy(float).reshape(ncl, 2) for g in genes}
    perm = np.random.RandomState(zlib.crc32(f"{raw}|{sim}".encode()) % 2**31).permutation(ncl); rows = []
    # for N in (3000, 6000):   # [2026-10-17 replaced: add the 1,500-twin design]
    for N in (1500, 3000, 6000):
        c = perm[:N]
        for i in genes:
            for j in genes:
                if j <= i or X[i][c].std() == 0 or X[j][c].std() == 0: continue
                S, Cv, zc = zC(X[i][c], X[j][c]); rows.append(dict(raw=raw, sim=sim, design=f"{N} twins", x=i, y=j, z_rho=zRho(X[i][c], X[j][c]), z_C=zc, z_U=zU(X[i][c], X[j][c]), S=S, C=Cv))
    return rows
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    F = files(); F = F[:a.limit] if a.limit else F; print(len(F), "simulation files", flush=True)
    import time; t = time.time()
    with mp.Pool(a.jobs) as p: res = p.map(one, F, chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(OUT if not a.limit else OUT.replace(".csv", "_test.csv"), index=False); print("done in %.0fs" % (time.time() - t))
