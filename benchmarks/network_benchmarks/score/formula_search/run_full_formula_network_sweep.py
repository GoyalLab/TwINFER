from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: these scripts used to sit in analysis_data/network_sweep_final and read/write their CSVs next to themselves; that data dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final'
ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/network_sweep_final"
TOPO_DIR = f"{ROOT}/input_data/network_sweep_final"

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_core import full_table_disjoint, compute_scores, score_topk
from benchmarks.network_benchmarks.score.formula_search.regdetector_vs_current import true_edges

N_GENES = 6
GENES = [f"gene_{i+1}" for i in range(N_GENES)]
T2 = 20


def main():
    topo_files = sorted(glob.glob(f"{TOPO_DIR}/grn_n6_*.txt"))
    rows = []
    t0 = time.time()
    for tf in topo_files:
        net = os.path.basename(tf)[:-4]
        true, poss = true_edges(tf)
        sims = sorted(glob.glob(f"{SIM_DIR}/df_{net}_rep*_*.csv"))
        for sim in sims:
            simrep = os.path.basename(sim).split(f"df_{net}_")[1].split("_")[0]
            for T1 in (1, 10):
                tsi = full_table_disjoint(sim, T1, T2, GENES)
                if tsi is None:
                    continue
                sc_all = compute_scores(tsi, GENES)
                for method in ("existdir", "full"):
                    mag = {p: v[method] for p, v in sc_all.items()}
                    sc = score_topk(mag, true, poss)
                    rows.append(dict(net=net, simrep=simrep, T1=T1, method=method, **sc))
        print(f"[{time.time()-t0:.0f}s] {net} ({len(sims)} reps)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/full_formula_network_sweep_results.csv", index=False)
    summ = df.groupby(["T1", "method"])[["auprc", "f1", "precision", "recall"]].mean()
    summ.to_csv(f"{HERE}/full_formula_network_sweep_summary.csv")
    print("\n=== network_sweep_final: existdir vs full (LARRY-tuned weights) ===")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
