"""Gated z_fanout: score(x,y) = existdir(x,y) + PENALTY if z_fanout(x,y) > THR, else
existdir(x,y) UNCHANGED. Tests whether only intervening when z_fanout confidently fires (rather
than a flat additive blend on every pair, or a standalone hard classifier) improves on existdir
alone. Uses the genuine permutation-based z_fanout (validated to work well in isolation).
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
from twinfer.scoring.analytic_core import full_table_disjoint, compute_scores
from benchmarks.network_benchmarks.score.formula_search.zhet_everywhere import true_edges

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
import twinfer.inference.correlation_functions as cf

N_SHUFFLES, N_CORES, SEED, THR = 200, 1, 101010, 2.326
PENALTIES = [-1.0, -2.0, -3.0, -5.0]

TOPO_DIR = f"{ROOT}/input_data/network_sweep_final"
SIM_DIR = f"{ROOT}/simulation_data/network_sweep_final"


def score_topk(mag, true, poss):
    from sklearn.metrics import auc, precision_recall_curve
    y = np.array([1 if p in true else 0 for p in poss], int)
    k = int(y.sum())
    if k == 0 or k == len(poss):
        return np.nan
    x = np.array([mag[p] for p in poss], float)
    prec, rec, _ = precision_recall_curve(y, x)
    return auc(rec, prec)


def main():
    topo_files = sorted(glob.glob(f"{TOPO_DIR}/grn_n6_*.txt"))
    rows = []
    t0 = time.time()
    for tf in topo_files:
        net = os.path.basename(tf)[:-4]
        true_e, poss = true_edges(tf)
        GENES = [f"gene_{i+1}" for i in range(6)]
        sims = sorted(f for f in glob.glob(f"{SIM_DIR}/df_{net}_rep*_*.csv")
                      if "simulation_before_division" not in os.path.basename(f))[:3]
        if not sims:
            continue
        for sim in sims:
            try:
                usecols = ["clone_id", "cell_id", "time_step", "replicate"] + [f"{g}_mRNA" for g in GENES]
                data = pd.read_csv(sim, usecols=usecols)
                views = select_views(data, 1, 20, seed=SEED)
                cross_z = calculate_all_cross_z(views, GENES, "clone", N_SHUFFLES, SEED, N_CORES, cf)
                tsi = full_table_disjoint(sim, 1, 20, GENES)
            except Exception as e:
                print("  skip", os.path.basename(sim), e, flush=True)
                continue
            if tsi is None:
                continue
            sc = compute_scores(tsi, GENES)
            existdir = {p: v["existdir"] for p, v in sc.items()}
            zfan = {p: calculate_z_fanout(p[0], p[1], GENES, cross_z)["z_fanout"] for p in poss}

            auprc_existdir = score_topk(existdir, true_e, poss)
            row = dict(net=net, sim=os.path.basename(sim), auprc_existdir=auprc_existdir)
            for pen in PENALTIES:
                gated = {p: existdir[p] + (pen if zfan[p] > THR else 0.0) for p in poss}
                row[f"auprc_gated_pen{pen}"] = score_topk(gated, true_e, poss)
            rows.append(row)
            del data, views, cross_z
            import gc; gc.collect()
        print(f"[{time.time()-t0:.0f}s] {net} ({len(sims)} sims)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/zfanout_gated_test_results.csv", index=False)
    print(f"\nwrote {len(df)} rows")
    print(df.mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
