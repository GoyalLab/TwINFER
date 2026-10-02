"""Ground truth for synthetic benchmarks: edge sets from a connectivity matrix file and the raw simulation files of one topology replicate.
[Moved from benchmarks/.../formula_search/full_formula_v2_common.py on 2026-09-30 after review. Matrix convention M[i, j] = effect of gene i on gene j, genes named gene_1..gene_n;
self-edges (the diagonal) are NOT part of the universe (project decision: self pairs are dropped when counting). zhet_everywhere.true_edges is the same function with dtype=int.]
"""
import glob
import os

import numpy as np


def true_edges(topo_path):
    """Dynamically sized from the matrix shape -- NOT hardcoded to N_GENES=6, so this
    works for both network_sweep_final (n=6) and mixed_network_sweep (n=6 or n=10)."""
    M = np.loadtxt(topo_path, delimiter=",", dtype=float)
    n = M.shape[0]
    genes = [f"gene_{i+1}" for i in range(n)]
    true = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    poss = [(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j]
    return true, poss, genes


def sim_files_for_topology(sim_dir, topology_name):
    """All raw simulation CSVs for one topology replicate, excluding pre-division
    snapshots (filename contains 'simulation_before_division')."""
    pattern = os.path.join(sim_dir, f"df_{topology_name}_rep*_*.csv")
    files = sorted(glob.glob(pattern))
    return [f for f in files if "simulation_before_division" not in os.path.basename(f)]
