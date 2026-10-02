"""Does z_fanout(x,y) = max_C min(z_abs_rho_t1(C,x), z_abs_rho_t1(C,y)) -- the strongest third
gene co-expressed with both x and y -- help on network_sweep_final (6-gene, densely/multiply
regulated), the same way it did on LARRY? Reuses full_table() from zscore_search_network_sweep.py
(already has z_abs_rho_t1 for every ordered pair -> the full symmetric co-expression-z matrix is
already present) and true_edges()/score() from regdetector_vs_current.py.

Scored per (topology, sim rep, T1) with AUPRC (full 30-pair universe) + top-k F1/precision/recall,
averaged, for s(|rho_t1|) + w*s(z_fanout) across a weight grid -- same style as one_score_search.py.
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
from benchmarks.network_benchmarks.score.formula_search.regdetector_vs_current import true_edges, score

N_GENES = 6
GENES = [f"gene_{i+1}" for i in range(N_GENES)]
WS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    return o


def zfanout_from_table(tab, poss):
    zmat = tab.set_index(["gene_1", "gene_2"]).z_abs_rho_t1.to_dict()
    zfan = np.zeros(len(poss))
    for k, (x, y) in enumerate(poss):
        cands = [g for g in GENES if g not in (x, y)]
        vals = []
        for c in cands:
            zcx = zmat.get((c, x), zmat.get((x, c)))
            zcy = zmat.get((c, y), zmat.get((y, c)))
            if zcx is not None and zcy is not None:
                vals.append(min(zcx, zcy))
        zfan[k] = max(vals) if vals else np.nan
    return zfan


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
                tab_i = tab.set_index(["gene_1", "gene_2"]).reindex(poss)
                abs_rho_t1 = tab_i.rho_t1.abs().to_numpy()
                if not np.isfinite(abs_rho_t1).all():
                    continue
                zfan = zfanout_from_table(tab, poss)
                if not np.isfinite(zfan).all():
                    continue
                sr, sz = s(abs_rho_t1), s(zfan)
                for w in WS:
                    mag = dict(zip(poss, sr + w * sz))
                    sc = score(mag, true, poss)
                    rows.append(dict(net=net, simrep=simrep, T1=T1, w=w, **sc))
        print(f"[{time.time()-t0:.0f}s] {net}  ({len(sims)} sim reps)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/zfanout_network_sweep_results.csv", index=False)
    print(f"\nwrote {HERE}/zfanout_network_sweep_results.csv  ({len(df)} rows)")

    summ = df.groupby(["T1", "w"])[["auprc", "f1", "precision", "recall"]].mean()
    summ.to_csv(f"{HERE}/zfanout_network_sweep_summary.csv")
    print("\n=== mean AUPRC / F1 / precision / recall, by t1 and weight w ===")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
