"""Flowchart statistics (step1_z, z_het, z_reg_gated; clone-weighted infer.py steps, 500 null draws) on the SAME clone subsamples as 24_fm01_twin_subsample_analytic.py, for a set of representative pairs.
Subsample = random fraction f of the clones with >=2 cells + all singleton cells (clone-size cap 50 as in script 23). f in 1.0 (1 run),0.7,0.5,0.3 (reps each).
usage: python 25_fm01_twin_subsample_flowchart.py OUT_CSV [--reps 6] [--jobs 14]"""
import os, sys, argparse, numpy as np, pandas as pd, anndata as ad, numba
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
from twinfer.utils.paths import get_data_root
ROOT = str(get_data_root().parent); sys.path.insert(0, f"{ROOT}/code/TwINFER/simulations/drift_multiple_state")
import run_6scenario_zscores as R
from twinfer.inference.correlation_functions import assign_twin_id, calculate_gated_regulation_statistic
R.N_SHUFFLES = 500
PAIRS = ["FOSL1-SERPINE1", "E2F1-UBE2T", "FOSL1-VEGFC", "FOSL1-TAGLN2", "FOSL1-AXL", "FOSL2-PRPSAP2", "HSF1-RPP25", "TFAP2A-PMEL", "TFAP2A-GYPC", "MITF-CITED1", "JUNB-CTHRC1", "TFAP2A-AKAP9"]
G = {}
def pick(bc, f, rep):      # same rule as script 24 (clone ids restricted to non-empty barcodes)
    n = bc.map(bc.value_counts()); ok = (bc.str.len() > 0) & (bc != "nan"); tw = np.array(sorted(bc[ok & (n >= 2)].unique())); rng = np.random.default_rng(1000 * int(round(f * 100)) + rep)
    keep = set(tw) if f >= 1 else set(rng.choice(tw, size=int(round(f * len(tw))), replace=False)); return ((ok & (n == 1)) | bc.isin(keep)).to_numpy()
def run(a):
    f, rep, pr = a; tf, g = pr.split("-", 1); df = G["df"]; m = pick(G["bc"], f, rep); d = df[m].copy()
    vc = d.clone_id.value_counts(); d = d[~d.clone_id.isin(vc[vc > 50].index)]
    d = pd.DataFrame({"clone_id": d.clone_id.to_numpy(), "cell_id": d.cell_id.to_numpy(), "time_step": 0, "gene_1_mRNA": d[tf].to_numpy(), "gene_2_mRNA": d[g].to_numpy()})
    R.GENES = ["gene_1", "gene_2"]; R.PAIR = ("gene_1", "gene_2"); numba.set_num_threads(1)
    with threadpool_limits(limits=1):
        tw = assign_twin_id(d[d.clone_id.map(d.clone_id.value_counts()) >= 2]).reset_index(drop=True); s1, _ = R._step1(d, 101011)
        s2, _, _ = R._step2(d, tw, 101010 + 271829, 101010 + 271832, 1); g3 = calculate_gated_regulation_statistic(tw, "gene_1", "gene_2", s2["z_het"], n_shuffles=500, random_state=101010 + 271840)
    return dict(f=f, rep=rep, pair=pr, N_t=int(tw.clone_id.nunique()), rho=s1["rho"], step1_z=s1["z"], z_het=s2["z_het"], rho_delta=s2["rho_delta"], z_reg_gated=g3["z_reg_gated"])
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--reps", type=int, default=6); ap.add_argument("--jobs", type=int, default=14); a = ap.parse_args()
    A = ad.read_h5ad(f"{ROOT}/finalized_data/FM01_data/FM01_integrated.h5ad"); genes = sorted({x for p in PAIRS for x in p.split("-", 1)})
    X = A.raw[:, genes].X; X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    bc = A.obs["fatemap_clone_singletcode"].astype(str).reset_index(drop=True); df = pd.DataFrame(X, columns=genes); df.insert(0, "cell_id", A.obs_names.to_numpy()); df.insert(0, "clone_id", bc.to_numpy())
    G["df"], G["bc"] = df, bc; del A
    tasks = [(f, r, p) for f in (1.0, 0.7, 0.5, 0.3) for r in (range(1) if f >= 1 else range(a.reps)) for p in PAIRS]; print(len(tasks), "runs", flush=True)
    with Pool(a.jobs) as pool: res = list(pool.imap_unordered(run, tasks, chunksize=1))
    pd.DataFrame(res).to_csv(a.out, index=False); print("done", len(res))
