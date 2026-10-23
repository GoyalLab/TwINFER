"""Figure 3 simulations (simulation_data/figure_3_1k, 2 genes, 4 regulation conditions, reps 0-99): per replicate z-scores / decomposition for
  X   = 1,500 twins + 3,000 singletons (figure2_1000_zscores.analyse: z_rho, z_C, z_U; same sampling as the figure-2 plots)
  N1500, N3000 = twin-only designs (twin_decomposition.twin_stats: also A_s = C/sigma_rho, A_u = U/sigma_rho = c z_U with z_rho = A_s + A_u).
Cache: <O>/twin_decomp_figure3.csv   usage: python figure3_decomposition.py [--jobs 8] [--reps 100]"""
import argparse, glob, os, numpy as np, pandas as pd
from figure2_1000_zscores import analyse, load, NCL, G
from twin_decomposition import twin_stats
from x_design_common import x_stats
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; SIM = A + "simulation_data/figure_3_1k/"
CONDS = ["A_rep_B", "A_rep_B_B_rep_A", "A_rep_B_B_to_A", "A_to_B_B_to_A"]
TAG = {"A_rep_B": "A_rep_B", "A_rep_B_B_rep_A": "A_and_B_both_repress", "A_rep_B_B_to_A": "A_rep_B_B_to_A", "A_to_B_B_to_A": "A_and_B"}      # label inside the file names (differs from the folder name for two conditions)
def rep_ids(cond):      # replicate ids present in the folder (not contiguous); newest file wins if an id is duplicated
    import re
    return sorted({int(re.search(rf"_ncells_6000_{TAG[cond]}_rep_(\d+)_", os.path.basename(f)).group(1)) for f in glob.glob(f"{SIM}{cond}/df_rows_*_ncells_6000_{TAG[cond]}_rep_*.csv")})
REPS = {c: rep_ids(c) for c in CONDS}
def find(cond, rep):
    fs = glob.glob(f"{SIM}{cond}/df_rows_*_ncells_6000_{TAG[cond]}_rep_{rep}_*.csv"); assert fs, (cond, rep); return sorted(fs, key=os.path.getmtime)[-1]
def one(rep):
    rows = []
    for si, cond in enumerate(CONDS):
        s = load(find(cond, REPS[cond][rep])); P1, P2 = s[1], s[20]      # rep = index 0..99 into the sorted replicate ids of this condition
        cc = np.random.RandomState(rep * 100 + si + 12345).permutation(NCL)          # one clone permutation per (condition, rep): N1500 = cc[:1500] (also the twins of X), N3000 = cc[:3000]
        o = x_stats(P2["gene_1"], P2["gene_2"], cc, np.random.RandomState(rep * 100 + si + 54321).randint(0, 2, 3000)); o.update(condition=cond, rep=rep, rep_id=REPS[cond][rep], design="X 1500 twins + 3000 singletons"); rows.append(o)   # [2026-10-17 was analyse() with its own partition]
        for N in (1500, 3000):
            t = twin_stats(P2["gene_1"][cc[:N]], P2["gene_2"][cc[:N]]); t.update(condition=cond, rep=rep, rep_id=REPS[cond][rep], design=f"N{N} twin pairs"); rows.append(t)
    return rows
if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--reps", type=int, default=100); a = ap.parse_args()
    with mp.Pool(a.jobs) as p: res = p.map(one, range(a.reps), chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(O + "twin_decomp_figure3.csv", index=False); print("figure3 done", flush=True)
