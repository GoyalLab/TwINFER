"""full_gated = existdir + (PENALTY if z_fanout > THR else 0) + 0.5*s(z_d_het)
   -- replaces full's flat 1.0*s(z_fanout) term with the gated version validated on
   network_sweep_final (existdir 0.635 -> gated 0.695 at penalty=-5). Genuine
   permutation-based z_fanout (calculate_all_cross_z / calculate_z_fanout), tested on
   mixed_network_sweep and the four real_data networks (GSD/HSC/VSC/mCAD).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: this script used to sit in the data dir analysis_data/paper_analysis/real_networks and read/write next to itself; that dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/analysis_data/network_sweep_final")
from paper_analysis.real_networks.fanout_mutual_z_boxplot import select_views, calculate_all_cross_z, calculate_z_fanout
from twinfer.scoring.analytic_core import full_table_disjoint, compute_scores, s
from benchmarks.network_benchmarks.score.formula_search.zhet_everywhere import true_edges

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
import twinfer.inference.correlation_functions as cf

N_SHUFFLES, N_CORES, SEED, THR = 200, 1, 101010, 2.326
PENALTIES = [-2.0]


def score_topk(mag, true, poss):
    from sklearn.metrics import auc, precision_recall_curve
    y = np.array([1 if p in true else 0 for p in poss], int)
    k = int(y.sum())
    if k == 0 or k == len(poss):
        return np.nan
    x = np.array([mag[p] for p in poss], float)
    prec, rec, _ = precision_recall_curve(y, x)
    return auc(rec, prec)


def process_one(sim, tf, true_e, poss, GENES, T1=1, T2=20):
    usecols = ["clone_id", "cell_id", "time_step", "replicate"] + [f"{g}_mRNA" for g in GENES]
    data = pd.read_csv(sim, usecols=usecols)
    views = select_views(data, T1, T2, seed=SEED)
    cross_z = calculate_all_cross_z(views, GENES, "clone", N_SHUFFLES, SEED, N_CORES, cf)
    tsi = full_table_disjoint(sim, T1, T2, GENES)
    if tsi is None:
        return None
    sc = compute_scores(tsi, GENES)
    existdir = {p: v["existdir"] for p, v in sc.items()}
    zfan = {p: calculate_z_fanout(p[0], p[1], GENES, cross_z)["z_fanout"] for p in poss}
    zdhet = tsi.reindex(poss).z_d_het.to_numpy()
    s_zdhet = dict(zip(poss, s(zdhet)))

    s_zfan = dict(zip(poss, s([zfan[p] for p in poss])))
    row = dict(auprc_existdir=score_topk(existdir, true_e, poss))
    full_flat = {p: existdir[p] + 1.0 * s_zfan[p] + 0.5 * s_zdhet[p] for p in poss}
    row["auprc_full_flat_perm"] = score_topk(full_flat, true_e, poss)
    for pen in PENALTIES:
        full_gated = {p: existdir[p] + (pen if zfan[p] > THR else 0.0) + 0.5 * s_zdhet[p] for p in poss}
        row[f"auprc_full_gated_pen{pen}"] = score_topk(full_gated, true_e, poss)
    del data, views, cross_z
    import gc; gc.collect()
    return row


def run_network_sweep_final(max_topos=15, max_reps=3):
    TOPO_DIR = f"{ROOT}/input_data/network_sweep_final"
    SIM_DIR = f"{ROOT}/simulation_data/network_sweep_final"
    rows = []
    t0 = time.time()
    for tf in sorted(glob.glob(f"{TOPO_DIR}/grn_n6_*.txt")):
        net = os.path.basename(tf)[:-4]
        GENES = [f"gene_{i+1}" for i in range(6)]
        true_e, poss = true_edges(tf)
        sims = sorted(f for f in glob.glob(f"{SIM_DIR}/df_{net}_rep*_*.csv")
                      if "simulation_before_division" not in os.path.basename(f))[:max_reps]
        if not sims:
            continue
        for sim in sims:
            try:
                r = process_one(sim, tf, true_e, poss, GENES)
            except Exception as e:
                print("  skip", os.path.basename(sim), e, flush=True)
                continue
            if r:
                r["dataset"] = "network_sweep_final"; r["net"] = net
                rows.append(r)
        print(f"[{time.time()-t0:.0f}s] nsf/{net} ({len(sims)} sims)", flush=True)
    return rows


def run_mixed_network_sweep(max_topos=10, max_reps=2):
    TOPO_DIR = f"{ROOT}/input_data/mixed_network_sweep"
    SIM_DIR = f"{ROOT}/simulation_data/mixed_network_sweep"
    rows = []
    t0 = time.time()
    for tf in sorted(glob.glob(f"{TOPO_DIR}/grn_n*_*.txt"))[:max_topos]:
        net = os.path.basename(tf)[:-4]
        M = np.loadtxt(tf, delimiter=",", dtype=int)
        n = M.shape[0]
        GENES = [f"gene_{i+1}" for i in range(n)]
        true_e, poss = true_edges(tf)
        sims = sorted(f for f in glob.glob(f"{SIM_DIR}/df_{net}_rep*_*.csv")
                      if "simulation_before_division" not in os.path.basename(f))[:max_reps]
        for sim in sims:
            try:
                r = process_one(sim, tf, true_e, poss, GENES)
            except Exception as e:
                print("  skip", os.path.basename(sim), e, flush=True)
                continue
            if r:
                r["dataset"] = "mixed_network_sweep"; r["net"] = net
                rows.append(r)
        print(f"[{time.time()-t0:.0f}s] mixed/{net} ({len(sims)} sims, n={n})", flush=True)
    return rows


def run_real_data(tok, topo_file, n, max_reps=6):
    TOPO_DIR = f"{ROOT}/input_data/real_world_networks"
    SIM_DIR = f"{ROOT}/simulation_data/real_data"
    tf = f"{TOPO_DIR}/{topo_file}"
    GENES = [f"gene_{i+1}" for i in range(n)]
    true_e, poss = true_edges(tf)
    sims = sorted(f for f in glob.glob(f"{SIM_DIR}/*_{tok}_*.csv")
                  if "simulation_before_division" not in os.path.basename(f))[:max_reps]
    rows = []
    t0 = time.time()
    for sim in sims:
        try:
            r = process_one(sim, tf, true_e, poss, GENES)
        except Exception as e:
            print("  skip", os.path.basename(sim), e, flush=True)
            continue
        if r:
            r["dataset"] = f"real_data:{tok}"; r["net"] = tok
            rows.append(r)
    print(f"[{time.time()-t0:.0f}s] real_data/{tok} ({len(sims)} sims, n={n})", flush=True)
    return rows


def main():
    rows = []
    rows += run_network_sweep_final(max_reps=3)
    rows += run_mixed_network_sweep(max_topos=10, max_reps=2)
    rows += run_real_data("GSD", "GSD.txt", 19, max_reps=5)
    rows += run_real_data("HSC_balanced", "HSC.txt", 11, max_reps=8)
    rows += run_real_data("VSC", "VSC.txt", 8, max_reps=8)
    rows += run_real_data("mCAD", "mCAD.txt", 5, max_reps=8)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/full_gated_fanout_results.csv", index=False)
    print(f"\nwrote {len(df)} rows")
    print(df.groupby("dataset").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
