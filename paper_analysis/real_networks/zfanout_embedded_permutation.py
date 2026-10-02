"""Does the GENUINE permutation-based z_fanout (the actual reference-script machinery,
identify_actual_directed_edges with real shuffles) correctly flag Fan-out-motif non-edges when
the motif is embedded in a larger network (not the isolated 3-gene case, where it was shown to
be essentially perfect)? Searches the true z_fanout definition -- max over ALL other genes in
the network, not just the triad's own source -- exactly as calculate_z_fanout already does.
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
from benchmarks.network_benchmarks.score.formula_search.find_motifs import find_motifs

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
import twinfer.inference.correlation_functions as cf

N_SHUFFLES = 200
N_CORES = 4
SEED = 101010


def run_dataset(name, topo_glob, sims_glob_fmt, max_reps_per_topo=5):
    rows = []
    t0 = time.time()
    for tf in sorted(glob.glob(topo_glob)):
        net = os.path.basename(tf)[:-4]
        M = np.loadtxt(tf, delimiter=",", dtype=int)
        n = M.shape[0]
        fan_out, _ = find_motifs(M)
        if not fan_out:
            continue
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
            for (a, b, c) in fan_out:
                gb, gc = GENES[b], GENES[c]
                res = calculate_z_fanout(gb, gc, GENES, cross_z)
                is_true = bool(M[b, c] != 0 or M[c, b] != 0)
                rows.append(dict(dataset=name, net=net, sim=os.path.basename(sim),
                                 pair=f"{gb},{gc}", z_fanout=res["z_fanout"], is_true_edge=is_true))
        print(f"[{time.time()-t0:.0f}s] {name}/{net} ({len(sims)} sims, {len(fan_out)} triads)", flush=True)
    return rows


def main():
    rows = []
    rows += run_dataset("network_sweep_final",
                        f"{ROOT}/input_data/network_sweep_final/grn_n6_*.txt",
                        lambda net: f"{ROOT}/simulation_data/network_sweep_final/df_{net}_rep*_*.csv",
                        max_reps_per_topo=5)
    rows += run_dataset("mixed_network_sweep",
                        f"{ROOT}/input_data/mixed_network_sweep/grn_n*_*.txt",
                        lambda net: f"{ROOT}/simulation_data/mixed_network_sweep/df_{net}_rep*_*.csv",
                        max_reps_per_topo=3)
    rows += run_dataset("real_data:GSD",
                        f"{ROOT}/input_data/real_world_networks/GSD.txt",
                        lambda net: f"{ROOT}/simulation_data/real_data/*_GSD_*.csv",
                        max_reps_per_topo=9)
    rows += run_dataset("real_data:HSC",
                        f"{ROOT}/input_data/real_world_networks/HSC.txt",
                        lambda net: f"{ROOT}/simulation_data/real_data/*_HSC_balanced_*.csv",
                        max_reps_per_topo=15)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/zfanout_embedded_permutation_results.csv", index=False)
    print(f"\nwrote {len(df)} rows")

    print("\n=== is high z_fanout actually a non-edge? per dataset ===")
    for name, g in df.groupby("dataset"):
        g = g[np.isfinite(g.z_fanout)]
        if g.empty:
            continue
        from sklearn.metrics import roc_auc_score, average_precision_score
        y_nonedge = (~g.is_true_edge).astype(int)
        try:
            auroc = roc_auc_score(y_nonedge, g.z_fanout)
            auprc = average_precision_score(y_nonedge, g.z_fanout)
        except Exception:
            auroc = auprc = float("nan")
        print(f"{name:22s} n={len(g):4d}  base_rate_nonedge={y_nonedge.mean():.3f}  "
              f"AUROC(high_zfan=>nonedge)={auroc:.3f}  AUPRC={auprc:.3f}")
        for thr in (2.326, 3.0):
            flagged = g.z_fanout > thr
            tp = (flagged & (y_nonedge == 1)).sum(); fp = (flagged & (y_nonedge == 0)).sum()
            fn = ((~flagged) & (y_nonedge == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            print(f"    thr={thr}: flagged={flagged.sum()}/{len(g)}  precision={prec:.3f}  recall={rec:.3f}")


if __name__ == "__main__":
    main()
