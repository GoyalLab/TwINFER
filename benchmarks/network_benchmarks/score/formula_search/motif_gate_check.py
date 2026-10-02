"""For every Fan-out and Feed-forward triad embedded in network_sweep_final, mixed_network_sweep,
and the real_data (GSD/HSC/mCAD/VSC) topologies, check whether the gated z_fanout rule validated
on figure_4 works: "own-pair |z_gamma|" (already in tsi, per-file calibrated) should be LOW for a
Fan-out confound pair (b,c) -- meaning it's safe to treat elevated z_fanout as evidence to
exclude -- and HIGH for a Feed-forward mediated pair (a,c) -- meaning elevated z_fanout there
must NOT be read as a confound, since it's a genuine (mediated) edge.

GATE_THR = 1.645 (same convention as z_reg_gated elsewhere in this codebase).
"works" per triad instance = the gate makes the STRUCTURALLY correct call:
  fan_out:      |z_gamma(b,c)| < GATE_THR   (confound pair correctly NOT flagged as a real edge)
  feed_forward: |z_gamma(a,c)| >= GATE_THR  (mediated real edge correctly NOT excluded)
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
# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: these scripts used to sit in analysis_data/network_sweep_final and read/write their CSVs next to themselves; that data dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_core import full_table_disjoint
from benchmarks.network_benchmarks.score.formula_search.find_motifs import find_motifs

GATE_THR = 1.645
T2 = 20


def process_dataset(name, topo_glob, sim_dir_fmt, sims_glob_fmt):
    rows = []
    t0 = time.time()
    topo_files = sorted(glob.glob(topo_glob))
    for tf in topo_files:
        net = os.path.basename(tf)[:-4]
        M = np.loadtxt(tf, delimiter=",", dtype=int)
        n = M.shape[0]
        GENES = [f"gene_{i+1}" for i in range(n)]
        fan_out, feed_forward = find_motifs(M)
        if not fan_out and not feed_forward:
            continue
        sims = sorted(f for f in glob.glob(sims_glob_fmt(net))
                      if "simulation_before_division" not in os.path.basename(f))
        for sim in sims:
            for T1 in (1, 10):
                tsi = full_table_disjoint(sim, T1, T2, GENES)
                if tsi is None:
                    continue
                for (a, b, c) in fan_out:
                    p = (GENES[b], GENES[c])
                    if p not in tsi.index:
                        continue
                    zg = abs(tsi.loc[p].z_gamma)
                    rows.append(dict(dataset=name, net=net, T1=T1, motif="fan_out",
                                     works=bool(zg < GATE_THR), zgamma=zg))
                for (a, b, c) in feed_forward:
                    p = (GENES[a], GENES[c])
                    if p not in tsi.index:
                        continue
                    zg = abs(tsi.loc[p].z_gamma)
                    rows.append(dict(dataset=name, net=net, T1=T1, motif="feed_forward",
                                     works=bool(zg >= GATE_THR), zgamma=zg))
        print(f"[{time.time()-t0:.0f}s] {name}/{net} ({len(sims)} sims)", flush=True)
    return rows


def main():
    all_rows = []
    all_rows += process_dataset(
        "network_sweep_final",
        f"{ROOT}/input_data/network_sweep_final/grn_n6_*.txt",
        None,
        lambda net: f"{ROOT}/simulation_data/network_sweep_final/df_{net}_rep*_*.csv",
    )
    all_rows += process_dataset(
        "mixed_network_sweep",
        f"{ROOT}/input_data/mixed_network_sweep/grn_n*_*.txt",
        None,
        lambda net: f"{ROOT}/simulation_data/mixed_network_sweep/df_{net}_rep*_*.csv",
    )
    for tok, topo in [("GSD", "GSD.txt"), ("HSC_balanced", "HSC.txt"), ("mCAD", "mCAD.txt"), ("VSC", "VSC.txt")]:
        all_rows += process_dataset(
            f"real_data:{tok}",
            f"{ROOT}/input_data/real_world_networks/{topo}",
            None,
            lambda net, tok=tok: (
                f"{ROOT}/simulation_data/real_data/*_{tok}_*.csv"
            ),
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{HERE}/motif_gate_check_results.csv", index=False)
    print(f"\nwrote {HERE}/motif_gate_check_results.csv  ({len(df)} rows)")

    print("\n=== fraction where the gate makes the correct call, by dataset x motif ===")
    summ = df.groupby(["dataset", "motif"]).works.agg(["mean", "count"])
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
