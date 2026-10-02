"""Recompute TwINFER's numbers for the mixed_network_sweep heatmap using THIS
SESSION's from-scratch 'full' formula (analytic_core_general.full_table_disjoint +
compute_scores), instead of the production package's shipped twinScore that used to
populate analysis_data/mixed_network_sweep/twinfer_analysis_output.csv.

48 topology replicates (16 "kinds" x 3 reps): kind = (n_genes, n_edges, cycle_len,
autoreg_frac), e.g. grn_n6_e6_c3_a0_pos50. T1=1, T2=20. Per-topology-replicate score
= mean AUPRC / F1-topk across whichever sim reps are available (3 here) for that
replicate. Competitor numbers are untouched: read straight from
mixed_network_sweep/beeline_analysis_output.csv, scheme=='spread', 'auprc'/'f1_topk'
(unsigned, directed), averaged per dataset_id across sim_rep.

Heatmap column grouping (unchanged from before): 8 columns = {n6,e6 / n6,e12 /
n10,e10 / n10,e20} x {Not autoregulated (a0) / Autoregulated (a3 for n6, a5 for n10)},
each column averaging over the 2 cycle_len variants x 3 topology reps that share that
(n, edges, autoreg-status) "kind".
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
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from benchmarks.network_benchmarks.score.formula_search.full_formula_v2_common import sim_files_for_topology, score_topology_replicate

ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/mixed_network_sweep"
TOPO_DIR = f"{ROOT}/input_data/mixed_network_sweep"
BEELINE_CSV = f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv"

ALGOS = ["GENIE3", "GRNBOOST2", "PPCOR", "SCODE", "SCSGL", "PEARSON", "PIDC"]

TOPOLOGY_NAMES = sorted(
    os.path.basename(f)[:-4] for f in glob.glob(f"{TOPO_DIR}/*.txt")
)


def compute_twinfer_full_scores():
    rows = []
    t0 = time.time()
    for topo_name in TOPOLOGY_NAMES:
        topo_txt = f"{TOPO_DIR}/{topo_name}.txt"
        sims = sim_files_for_topology(SIM_DIR, topo_name)
        r = score_topology_replicate(sims, topo_txt)
        rows.append(dict(topology=topo_name, n_sim_reps=r["n_sim_reps"],
                          auprc=r["auprc"], f1_topk=r["f1"]))
        print(f"[{time.time()-t0:.0f}s] {topo_name}: n_sims={r['n_sim_reps']} "
              f"auprc={r['auprc']:.3f} f1={r['f1']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/full_formula_v2_mixed_network_sweep_twinfer.csv", index=False)
    print(f"\nWrote {HERE}/full_formula_v2_mixed_network_sweep_twinfer.csv")
    return df


def load_competitor_scores():
    df = pd.read_csv(BEELINE_CSV)
    df = df[df.scheme == "spread"]
    g = df.groupby(["dataset_id", "algorithm"], as_index=False)[["auprc", "f1_topk"]].mean()
    g.to_csv(f"{HERE}/full_formula_v2_mixed_network_sweep_competitors.csv", index=False)
    print(f"Wrote {HERE}/full_formula_v2_mixed_network_sweep_competitors.csv")
    return g


def main():
    compute_twinfer_full_scores()
    load_competitor_scores()


if __name__ == "__main__":
    main()
