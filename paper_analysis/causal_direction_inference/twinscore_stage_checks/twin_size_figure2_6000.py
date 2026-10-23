"""z_rho, z_C, z_U for the figure 2 two-gene scenarios with ALL 6,000 clones as twin pairs (12,000 cells), 100 replicates; same sampling code as twin_size_scan.py (NS overridden).  Output: <O>/twin_size_scan_figure2_6000.csv"""
import os, pandas as pd, multiprocessing as mp
import twin_size_scan as S
S.NS = [6000]
if __name__ == "__main__":
    out = S.O + "twin_size_scan_figure2_6000.csv"
    if not os.path.exists(out):
        with mp.Pool(8) as p: res = p.map(S.fig2_rep, range(100), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(out, index=False); print("done", flush=True)
