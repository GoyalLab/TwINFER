# %%
"""
Seeded-initial-condition variant of real_network_multistate_sim.py.

Instead of every cell starting empty, ~n_cells/k cells are initialised in each of
the network's k stable basins (mean-field fixed points), so the population is
BALANCED across states rather than all draining into one basin.

Same per-network (Hill n, k_add) as the IC-unset run. 3 reps, 2000 before-division
steps (VSC 6000). SLURM array config_index layout matches the IC-unset script:
    0-2 VSC | 3-5 mCAD | 6-8 GSD | 9-11 HSC | 12-14 EMT

Output: <data_root>/paper_analysis/<net>/simulate/<run_tag>_seeded/
"""
import argparse
import os
import sys

import numpy as np
from numba import get_num_threads, set_num_threads

sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code")  # predict_multistability
from predict_multistability import MeanField, load_params

from twinfer.simulation.gillespie_simulations import process_param_set, read_input_matrix
from twinfer.utils.paths import get_data_root, stage_dir

from real_network_multistate_sim import NETWORKS, N_REPS, PLAN, INPUT_ROOT, PARAM_CSV, build_matrices

# cells start AT the fixed points, so no long relaxation is needed -- a short
# before-division window is enough to reach the stochastic stationary spread.
SEEDED_SIM_TIME = {"VSC": 1500}
SEEDED_SIM_TIME_DEFAULT = 800


def seeded_states(M, n_hill, kadd_scale):
    """(k, n_genes) arrays of mean-field (mRNA*, protein*, active_frac) for each
    STABLE fixed point, ordered largest-mean-expression first."""
    P = load_params()
    mf = MeanField(M, P, kadd_scale=kadd_scale, n_hill=n_hill)
    fps = [f for f in mf.fixed_points(n_starts=250) if f["stable"]]
    mrna = np.array([P["kpm"] * f["a"] / P["kdm"] for f in fps])
    prot = np.array([f["p"] for f in fps])
    afrac = np.array([f["a"] for f in fps])
    return mrna, prot, afrac


def make_pop0_builder(mrna, prot, afrac):
    """Return f(species_index, gene_list, n_cells) -> (n_species, n_cells) int pop0_mat
    with n_cells split as evenly as possible across the k seed states."""
    k = len(prot)

    def build(species_index, gene_list, n_cells):
        n_species = len(species_index)
        pop = np.zeros((n_species, n_cells), dtype=np.int64)
        # block assignment of cells to states
        bounds = np.linspace(0, n_cells, k + 1).astype(int)
        for s in range(k):
            lo, hi = bounds[s], bounds[s + 1]
            for gi, g in enumerate(gene_list):
                a = 1 if afrac[s, gi] > 0.5 else 0
                pop[species_index[f"{g}_A"], lo:hi] = a
                pop[species_index[f"{g}_I"], lo:hi] = 1 - a
                pop[species_index[f"{g}_mRNA"], lo:hi] = int(round(max(mrna[s, gi], 0)))
                pop[species_index[f"{g}_protein"], lo:hi] = int(round(max(prot[s, gi], 0)))
        return pop

    return build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_index", type=int, default=0)
    args, _ = ap.parse_known_args()
    idx = args.config_index
    if not (0 <= idx < len(PLAN)):
        raise ValueError(f"config_index {idx} out of range 0..{len(PLAN) - 1}")

    net, rep = PLAN[idx]
    spec = dict(NETWORKS[net])
    spec["sim_time"] = SEEDED_SIM_TIME.get(net, SEEDED_SIM_TIME_DEFAULT)
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    set_num_threads(cores)

    matrix_path = f"{INPUT_ROOT}/real_world_networks/{net}.txt"
    n_genes, M = read_input_matrix(matrix_path)
    n_matrix, k_add_matrix = build_matrices(M, spec["n_hill"], spec["kadd_scale"])
    mrna, prot, afrac = seeded_states(M, spec["n_hill"], spec["kadd_scale"])
    k = len(prot)

    run_tag = f"{spec['run_tag']}_seeded"
    out_dir = stage_dir(net, "simulate", run_tag=run_tag, make_latest=False)
    os.makedirs(f"{out_dir}/logs", exist_ok=True)

    print(f"[config {idx}] net={net} rep={rep} genes={n_genes} n_hill={spec['n_hill']} "
          f"kadd_scale={spec['kadd_scale']} sim_time={spec['sim_time']} "
          f"seed_states={k} threads={get_num_threads()}", flush=True)
    print(f"  data_root={get_data_root()}  out_dir={out_dir}", flush=True)
    for s in range(k):
        on = [gi + 1 for gi in range(n_genes) if afrac[s, gi] > 0.5]
        print(f"  seed state {s}: {len(on)}/{n_genes} genes ON  (gene idx {on})", flush=True)

    cfg = {
        "n_cells": 6000,
        "simulation_time_before_division": spec["sim_time"],
        "twin_simulation_time_after_division": 48,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": matrix_path,
        "param_csv": PARAM_CSV,
        "rows_to_use": [[0] * n_genes],
        "output_folder": str(out_dir),
        "log_file": f"{out_dir}/logs/{net}.log",
        "type": f"{net}_seeded_rep_{rep}",
        "combinatorial_interaction_type": "additive",
        "n_matrix": n_matrix,
        "k_add_matrix": k_add_matrix,
        "use_csv_k_add": False,
        "pop0_mat": make_pop0_builder(mrna, prot, afrac),
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": cores,
    }
    label = "rows_" + "_".join(map(str, cfg["rows_to_use"][0]))
    path = process_param_set(cfg["rows_to_use"][0], label, cfg)
    print(f"[config {idx}] DONE -> {path}", flush=True)


if __name__ == "__main__":
    main()
