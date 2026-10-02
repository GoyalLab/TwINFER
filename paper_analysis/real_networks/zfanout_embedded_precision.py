"""Precision check for the genuine permutation-based z_fanout on embedded networks: for EVERY
directed pair (not just the Fan-out-motif triads), does thresholding z_fanout > 2.326 correctly
separate true edges from non-edges? This is the precision side missing from
zfanout_embedded_permutation.py, which only tested recall on the guaranteed-non-edge triad pairs.
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
from benchmarks.network_benchmarks.score.formula_search.zhet_everywhere import true_edges

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
import twinfer.inference.correlation_functions as cf

N_SHUFFLES = 200
N_CORES = 4
SEED = 101010
THR = 2.326


def run_dataset(name, topo_glob, sims_glob_fmt, max_reps_per_topo=3, max_topos=None):
    rows = []
    t0 = time.time()
    topo_files = sorted(glob.glob(topo_glob))
    if max_topos:
        topo_files = topo_files[:max_topos]
    for tf in topo_files:
        net = os.path.basename(tf)[:-4]
        true_e, poss = true_edges(tf)
        n = len(set(g for p in poss for g in p))
        GENES = [f"gene_{i+1}" for i in range(n)]
        sims = sorted(f for f in glob.glob(sims_glob_fmt(net))
                      if "simulation_before_division" not in os.path.basename(f))[:max_reps_per_topo]
        for sim in sims:
            try:
                data = pd.read_csv(sim)
                views = select_views(data, 1, 20, seed=SEED)
                cross_z = calculate_all_cross_z(views, GENES, "clone", N_SHUFFLES, SEED, N_CORES, cf)
            except Exception as e:
                print(f"  skip {os.path.basename(sim)}: {e}", flush=True)
                continue
            for (gx, gy) in poss:
                res = calculate_z_fanout(gx, gy, GENES, cross_z)
                rows.append(dict(dataset=name, net=net, sim=os.path.basename(sim),
                                 pair=f"{gx},{gy}", z_fanout=res["z_fanout"],
                                 is_true_edge=(gx, gy) in true_e))
        print(f"[{time.time()-t0:.0f}s] {name}/{net} ({len(sims)} sims, {len(poss)} pairs)", flush=True)
    return rows


def main():
    rows = []
    rows += run_dataset("network_sweep_final",
                        f"{ROOT}/input_data/network_sweep_final/grn_n6_*.txt",
                        lambda net: f"{ROOT}/simulation_data/network_sweep_final/df_{net}_rep*_*.csv",
                        max_reps_per_topo=3, max_topos=8)
    rows += run_dataset("mixed_network_sweep",
                        f"{ROOT}/input_data/mixed_network_sweep/grn_n*_*.txt",
                        lambda net: f"{ROOT}/simulation_data/mixed_network_sweep/df_{net}_rep*_*.csv",
                        max_reps_per_topo=2, max_topos=10)
    rows += run_dataset("real_data:GSD",
                        f"{ROOT}/input_data/real_world_networks/GSD.txt",
                        lambda net: f"{ROOT}/simulation_data/real_data/*_GSD_*.csv",
                        max_reps_per_topo=6)
    rows += run_dataset("real_data:HSC",
                        f"{ROOT}/input_data/real_world_networks/HSC.txt",
                        lambda net: f"{ROOT}/simulation_data/real_data/*_HSC_balanced_*.csv",
                        max_reps_per_topo=10)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/zfanout_embedded_precision_results.csv", index=False)
    print(f"\nwrote {len(df)} rows")

    print(f"\n=== precision/recall at z_fanout > {THR}, all pairs (not just fan-out triads) ===")
    for name, g in df.groupby("dataset"):
        g = g[np.isfinite(g.z_fanout)]
        y = g.is_true_edge.to_numpy()
        z = g.z_fanout.to_numpy()
        flagged_nonedge = z > THR  # "flag this pair as a likely confound / non-edge"
        tp = (flagged_nonedge & ~y).sum()  # correctly flagged non-edge
        fp = (flagged_nonedge & y).sum()   # WRONGLY flagged a true edge as confound
        fn = ((~flagged_nonedge) & ~y).sum()
        tn = ((~flagged_nonedge) & y).sum()
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"{name:22s} n={len(g):5d}  n_true_edges={y.sum():4d}  base_rate_true={y.mean():.3f}  "
              f"flagged_as_nonedge={flagged_nonedge.sum():4d}  "
              f"correctly_flagged_nonedge={tp:4d}  wrongly_flagged_true_edge={fp:4d}  "
              f"precision(of flags, frac actually non-edge)={precision:.3f}  "
              f"recall(of non-edges, frac caught)={recall:.3f}")


if __name__ == "__main__":
    main()
