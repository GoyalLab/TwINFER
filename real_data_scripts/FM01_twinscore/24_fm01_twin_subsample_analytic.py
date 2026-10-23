"""Subsample the FM01 twin clones and re-score the 200 pairs (top 100 by |C| and by |U|) with the analytic TwinScore statistics (rho, S, C, U, z_rho, z_C(h), z_U, sigma from the
subsample's own clone sizes) via twinscore_pdf_candidates.py (--sample-col trick on a copied qc folder).
Subsample = random fraction f of the clones with >=2 cells (all their cells kept) + ALL singleton cells. f in 1.0,0.7,0.5,0.3; reps per f (default 20).
usage: python 24_fm01_twin_subsample_analytic.py OUT_DIR [--reps 20] [--jobs 12]   -> OUT_DIR/subsample_analytic.csv"""
import os, sys, argparse, subprocess, shutil, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; QC = f"{ROOT}/finalized_data/FM01_data/qc_filtered"
PY = "/home/gzu5140/.conda/envs/twinfer-code/bin/python"; SCRIPT = f"{ROOT}/code/TwINFER/real_data_scripts/twinscore_pdf_candidates.py"
T = f"{ROOT}/analysis_data/fm01/data/top100_C_U_scenarios"
def subset(obs, f, rep):
    bc = obs["fatemap_clone_singletcode"].astype(str); n = bc.map(bc.value_counts()); ok = (bc.str.len() > 0) & (bc != "nan")
    tw = np.array(sorted(bc[ok & (n >= 2)].unique())); rng = np.random.default_rng(1000 * int(round(f * 100)) + rep)
    keep = set(tw) if f >= 1 else set(rng.choice(tw, size=int(round(f * len(tw))), replace=False))
    return ((ok & (n == 1)) | bc.isin(keep)).astype(int)
def run(a):
    out, f, rep, cand = a; tag = f"_f{int(round(f*100))}_r{rep}"; d = f"{out}/work{tag}"; os.makedirs(d, exist_ok=True)
    for fn in ("fm01_qc_counts.mtx", "genes.txt"):
        if not os.path.exists(f"{d}/{fn}"): os.symlink(f"{QC}/{fn}", f"{d}/{fn}")
    obs = pd.read_csv(f"{QC}/obs_metadata.csv", index_col=0, dtype={"replicate": str}); obs["sub"] = subset(obs, f, rep).to_numpy(); obs.to_csv(f"{d}/obs_metadata.csv")
    env = dict(os.environ, TWINFER_PROJECT_ROOT=ROOT, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    fn = f"{d}/pdf_procedure_all_candidate_pairs{tag}.csv"
    for attempt in range(4):                       # scipy's parallel mtx reader fails intermittently when many processes open the file at once
        r = subprocess.run([PY, SCRIPT, "--dataset", "FM01", "--candidates", cand, "--out", d, "--qc-dir", d, "--sample-col", "sub", "--sample-value", "1", "--tag", tag], env=env, capture_output=True, text=True)
        if os.path.exists(fn): break
    if not os.path.exists(fn): return None
    t = pd.read_csv(fn); import json; s = json.load(open(f"{d}/summary{tag}.json")); t["f"] = f; t["rep"] = rep; t["N_t"] = s["N_t"]
    shutil.rmtree(d, ignore_errors=True); return t
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    p = pd.concat([pd.read_csv(f"{T}/top100_by_absC_scenarios.csv"), pd.read_csv(f"{T}/top100_by_absU_scenarios.csv")]).drop_duplicates("pair")
    cand = f"{a.out}/pairs200.tsv"; p.pair.str.split("-", n=1, expand=True).to_csv(cand, sep="\t", header=False, index=False)
    prev = pd.read_csv(f"{a.out}/subsample_analytic.csv") if os.path.exists(f"{a.out}/subsample_analytic.csv") else pd.DataFrame(columns=["f", "rep"])
    have = set(zip(prev.f.round(2), prev.rep)); tasks = [(a.out, f, r, cand) for f in (1.0, 0.7, 0.5, 0.3) for r in (range(1) if f >= 1 else range(a.reps)) if (round(f, 2), r) not in have]; print(len(tasks), "runs to do,", len(have), "already done", flush=True)
    with Pool(a.jobs) as pool: res = [x for x in pool.imap_unordered(run, tasks) if x is not None]
    pd.concat([prev] + res).to_csv(f"{a.out}/subsample_analytic.csv", index=False); print("done", len(res), "new runs")
