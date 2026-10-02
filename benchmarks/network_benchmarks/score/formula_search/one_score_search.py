"""Find ONE combined score (same formula, no t1-dependent switching) that works well at
BOTH t1=1h and t1=10h on network_sweep_final. Reuses full_table() from
zscore_search_network_sweep.py (per-pair raw quantities) and true_edges()/score() from
regdetector_vs_current.py (AUPRC + top-k F1/precision/recall), scored per (topology, sim
rep, T1), then averaged.

Candidates tested (all standardized with s() over the replicate's own 30-pair panel,
sign chosen by AUPRC on t1=1h and held fixed):
  zg          : s(|z_gamma|)
  cross       : s(abs_rho_cross_xy)
  zg_cross    : s(|z_gamma|) + s(abs_rho_cross_xy)
  zg_coexp    : s(|z_gamma|) + s(|rho_t2|)
  zg_cross_coexp : s(|z_gamma|) + s(abs_rho_cross_xy) + s(|rho_t2|)
  current     : shipped twinScore (baseline, from regdetector_vs_current.py's own calc)

Output: one_score_search_results.csv, one_score_search_summary.csv
"""
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
from benchmarks.network_benchmarks.score.formula_search.zscore_search_network_sweep import full_table
from benchmarks.network_benchmarks.score.formula_search.regdetector_vs_current import true_edges, score, analytic_for_file


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    return o


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
                tab = full_table(sim, T1)
                if tab is None:
                    continue
                tab = tab.set_index(["gene_1", "gene_2"]).reindex(poss)

                zg = s(np.abs(tab.z_gamma.to_numpy()))
                cross = s(tab.abs_rho_cross_xy.to_numpy())
                coexp = s(np.abs(tab.rho_t2.to_numpy()))

                cands = {
                    "zg": zg,
                    "cross": cross,
                    "zg_cross": zg + cross,
                    "zg_coexp": zg + coexp,
                    "zg_cross_coexp": zg + cross + coexp,
                }
                for name, vals in cands.items():
                    mag = dict(zip(poss, vals))
                    sc = score(mag, true, poss)
                    rows.append(dict(net=net, simrep=simrep, T1=T1, method=name, **sc))

                res = analytic_for_file(sim, T1)
                if res is not None:
                    current, _ = res
                    sc = score(current, true, poss)
                    rows.append(dict(net=net, simrep=simrep, T1=T1, method="current", **sc))
        print(f"[{time.time()-t0:.0f}s] {net}  ({len(sims)} sim reps)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/one_score_search_results.csv", index=False)
    print(f"\nwrote {HERE}/one_score_search_results.csv  ({len(df)} rows)")

    summ = df.groupby(["T1", "method"])[["auprc", "f1", "precision", "recall"]].mean()
    summ.to_csv(f"{HERE}/one_score_search_summary.csv")
    print("\n=== mean AUPRC / F1 / precision / recall, by t1 and method ===")
    print(summ.round(3).to_string())

    piv = df.groupby(["method", "T1"])["auprc"].mean().unstack("T1")
    piv["min_both"] = piv.min(axis=1)
    piv["avg_both"] = piv.mean(axis=1)
    piv = piv.sort_values("min_both", ascending=False)
    print("\n=== AUPRC at t1=1 vs t1=10, ranked by worst-case (min) across both ===")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
