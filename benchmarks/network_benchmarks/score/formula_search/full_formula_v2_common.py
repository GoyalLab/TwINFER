"""Shared helpers for the corrected 'full' formula heatmap recompute (this session's
own full(x->y) = s(|rho_t1|)+s(|rho_t2|)+s(direction)+1.0*s(z_fanout)+0.5*s(z_d_het),
via analytic_core_general.full_table_disjoint + compute_scores), replacing the
production-package twinScore numbers that used to populate these heatmaps.

Used by both network_sweep_final and mixed_network_sweep heatmap rebuilds.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_core import full_table_disjoint, compute_scores, score_topk

T1_DEFAULT = 1
T2_DEFAULT = 20


from twinfer.scoring.benchmark_truth import true_edges, sim_files_for_topology
# [2026-09-30 moved to twinfer.scoring (benchmark_truth): true_edges commented out here]
# def true_edges(topo_path):
#     """Dynamically sized from the matrix shape -- NOT hardcoded to N_GENES=6, so this
#     works for both network_sweep_final (n=6) and mixed_network_sweep (n=6 or n=10)."""
#     M = np.loadtxt(topo_path, delimiter=",", dtype=float)
#     n = M.shape[0]
#     genes = [f"gene_{i+1}" for i in range(n)]
#     true = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
#     poss = [(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j]
#     return true, poss, genes


# [2026-09-30 moved to twinfer.scoring (benchmark_truth): sim_files_for_topology commented out here]
# def sim_files_for_topology(sim_dir, topology_name):
#     """All raw simulation CSVs for one topology replicate, excluding pre-division
#     snapshots (filename contains 'simulation_before_division')."""
#     pattern = os.path.join(sim_dir, f"df_{topology_name}_rep*_*.csv")
#     files = sorted(glob.glob(pattern))
#     return [f for f in files if "simulation_before_division" not in os.path.basename(f)]


# Shared column-group definitions for the network_sweep_final heatmap (density family +
# sign-fraction family, sharing the 9-edge/all-positive topology replicates -- see
# full_formula_v2_network_sweep.py's module docstring for the full rationale).
NETWORK_SWEEP_STANDARD_FAMILIES = [
    ("5 edges", "grn_n6_e5_pos100_density"),
    ("9 edges", "grn_n6_e9_pos100_center"),
    ("17 edges", "grn_n6_e17_pos100_density"),
    ("all-neg", "grn_n6_e9_pos0_sign_ratio"),
    ("balanced", "grn_n6_e9_pos50_sign_ratio"),
    ("all-pos", "grn_n6_e9_pos100_center"),  # duplicate of "9 edges" -- same data, on purpose
]
NETWORK_SWEEP_E13_TAG_TO_REP = {"rep3": 0, "rep1": 1, "rep2": 2}
NETWORK_SWEEP_GROUP_ORDER = ["5 edges", "9 edges", "13 edges", "17 edges", "all-neg", "balanced", "all-pos"]


def network_sweep_group_col_to_topology():
    """(group_label, col 1/2/3) -> topology name, matching full_formula_v2_network_sweep.py."""
    mapping = {}
    for group_label, prefix in NETWORK_SWEEP_STANDARD_FAMILIES:
        for rep_idx in (0, 1, 2):
            mapping[(group_label, rep_idx + 1)] = f"{prefix}_rep{rep_idx}"
    for tag, rep_idx in NETWORK_SWEEP_E13_TAG_TO_REP.items():
        mapping[("13 edges", rep_idx + 1)] = f"grn_n6_e13_pos100_center_rep{rep_idx}"
    return mapping


def score_topology_replicate(sim_csvs, topo_txt_path, T1=T1_DEFAULT, T2=T2_DEFAULT, seed=0):
    """Runs the fresh 'full' formula on every sim rep CSV for one topology replicate,
    scores each against that replicate's own true edges, and averages AUPRC/F1-topk
    (plain mean) across whichever sim reps are available."""
    true, poss, genes = true_edges(topo_txt_path)
    aurocs, f1s, n_ok = [], [], 0
    for csv_path in sim_csvs:
        tsi = full_table_disjoint(csv_path, T1, T2, genes, seed=seed)
        if tsi is None:
            continue
        sc_all = compute_scores(tsi, genes)
        mag = {p: v["full"] for p, v in sc_all.items()}
        sc = score_topk(mag, true, poss)
        if np.isnan(sc["auprc"]) and np.isnan(sc["f1"]):
            continue
        aurocs.append(sc["auprc"])
        f1s.append(sc["f1"])
        n_ok += 1
    if n_ok == 0:
        return dict(auprc=np.nan, f1=np.nan, n_sim_reps=0)
    return dict(auprc=float(np.nanmean(aurocs)), f1=float(np.nanmean(f1s)), n_sim_reps=n_ok)
