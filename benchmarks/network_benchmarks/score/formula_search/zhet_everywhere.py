"""z_het (best sign per file) tested across every synthetic/real network dataset used this
session: network_sweep_final, mixed_network_sweep, and the four real_data networks
(GSD/HSC/mCAD/VSC). LARRY's own per-gene-set z_het numbers were already computed earlier
(mean AUPRC 0.190, backwards on 7/9 gene sets) and are not recomputed here.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

ROOT = f'{TWINFER_PROJECT_ROOT}'
# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: these scripts used to sit in analysis_data/network_sweep_final and read/write their CSVs next to themselves; that data dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_core import full_table_disjoint

T2 = 20


def true_edges(topo_path):
    """Dynamically sized -- do not import the N_GENES=6-hardcoded version from
    regdetector_vs_current for datasets with variable gene counts (it silently truncates
    10-gene mixed_network_sweep topologies and crashes on mCAD's 5 genes)."""
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    n = M.shape[0]
    genes = [f"gene_{i+1}" for i in range(n)]
    true = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    poss = [(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j]
    return true, poss


def best_sign_auprc(y, x):
    best = None
    for sgn in (1, -1):
        prec, rec, _ = precision_recall_curve(y, sgn * x)
        au = auc(rec, prec)
        if best is None or au > best[1]:
            best = (sgn, au)
    return best


def score_zhet(true_e, poss, tab):
    y_full = np.array([1 if p in true_e else 0 for p in poss], int)
    zhet_full = tab.reindex(poss).z_het.to_numpy()
    finite = np.isfinite(zhet_full)
    if finite.sum() < 4:
        return None
    y, zhet = y_full[finite], zhet_full[finite]
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return best_sign_auprc(y, zhet)


def run_topology_dataset(name, topo_glob, sim_glob_fmt, n_genes_from_matrix=True):
    rows = []
    t0 = time.time()
    for tf in sorted(glob.glob(topo_glob)):
        net = os.path.basename(tf)[:-4]
        true_e, poss = true_edges(tf)
        n = len(set(g for p in poss for g in p))
        GENES = [f"gene_{i+1}" for i in range(n)]
        sims = sorted(glob.glob(sim_glob_fmt(net)))
        for sim in sims:
            for T1 in (1, 10):
                tsi = full_table_disjoint(sim, T1, T2, GENES)
                if tsi is None:
                    continue
                r = score_zhet(true_e, poss, tsi)
                if r is None:
                    continue
                rows.append(dict(dataset=name, net=net, T1=T1, sign=r[0], auprc=r[1]))
        print(f"[{time.time()-t0:.0f}s] {name}/{net} ({len(sims)} sims)", flush=True)
    return rows


def run_real_data(tok, topo_file, sim_dir_override=None):
    tf = f"{ROOT}/input_data/real_world_networks/{topo_file}"
    true_e, poss = true_edges(tf)
    n = len(set(g for p in poss for g in p))
    GENES = [f"gene_{i+1}" for i in range(n)]
    sdir = sim_dir_override or f"{ROOT}/simulation_data/real_data"
    sims = sorted(f for f in glob.glob(f"{sdir}/*_{tok}_*.csv")
                  if "simulation_before_division" not in os.path.basename(f))
    rows = []
    t0 = time.time()
    for sim in sims:
        for T1 in (1, 10):
            tsi = full_table_disjoint(sim, T1, T2, GENES)
            if tsi is None:
                continue
            r = score_zhet(true_e, poss, tsi)
            if r is None:
                continue
            rows.append(dict(dataset=f"real_data:{tok}", net=tok, T1=T1, sign=r[0], auprc=r[1]))
    print(f"[{time.time()-t0:.0f}s] real_data/{tok} ({len(sims)} sims)", flush=True)
    return rows


def main():
    rows = []
    rows += run_topology_dataset("network_sweep_final",
                                  f"{ROOT}/input_data/network_sweep_final/grn_n6_*.txt",
                                  lambda net: f"{ROOT}/simulation_data/network_sweep_final/df_{net}_rep*_*.csv")
    rows += run_topology_dataset("mixed_network_sweep",
                                  f"{ROOT}/input_data/mixed_network_sweep/grn_n*_*.txt",
                                  lambda net: f"{ROOT}/simulation_data/mixed_network_sweep/df_{net}_rep*_*.csv")
    EXTRA_SIM_DIRS = {
        "EMT": f"{ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653",
        "B_cell_activation": f"{ROOT}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622",
        "Pluripotent": f"{ROOT}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511",
    }
    for tok, topo in [("GSD", "GSD.txt"), ("HSC_balanced", "HSC.txt"), ("mCAD", "mCAD.txt"), ("VSC", "VSC.txt"),
                      ("EMT", "EMT.txt"), ("B_cell_activation", "B_cell.txt"), ("Pluripotent", "Pluripotent.txt")]:
        rows += run_real_data(tok, topo, sim_dir_override=EXTRA_SIM_DIRS.get(tok))

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/zhet_everywhere_results.csv", index=False)
    print(f"\nwrote {HERE}/zhet_everywhere_results.csv  ({len(df)} rows)")
    summ = df.groupby(["dataset", "T1"]).agg(mean_auprc=("auprc", "mean"),
                                              mode_sign=("sign", lambda x: int(np.sign(x.mean()) or 1)),
                                              n=("auprc", "size"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
