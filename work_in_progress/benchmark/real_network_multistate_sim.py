# %%
"""
Re-run the TwINFER Gillespie sims for the real networks that CAN be pushed into a
multi-stable regime, using per-network (Hill n, k_add) tuned so the mean-field
limit holds >= 2 well-separated stable fixed points.

This is the DEFAULT-initial-condition ("IC unset") variant: every cell starts
from the empty state, exactly as the standard sims do. Expect the population to
land mostly in one basin (basin volumes are unequal) -- the seeded-IC variant is
what gives balanced occupancy.

3 replicates per network. Before-division (steady-state) time = 2000 steps,
except VSC = 6000 steps (its 5-way structure develops slowly).

One rep per --config_index for a SLURM array (see NETWORKS for the (n, k_add)):
    0,1,2   -> VSC   rep 0,1,2   (6000 steps)
    3,4,5   -> mCAD  rep 0,1,2   (2000 steps)
    6,7,8   -> GSD   rep 0,1,2   (2000 steps)
    9,10,11 -> HSC   rep 0,1,2   (2000 steps)
    12,13,14-> EMT   rep 0,1,2   (2000 steps)

Output: <data_root>/paper_analysis/<net>/simulate/<run_tag>/
"""
import argparse
import os

import numpy as np
from numba import get_num_threads, set_num_threads

from twinfer.simulation.gillespie_simulations import process_param_set, read_input_matrix
from twinfer.utils.paths import get_data_root, stage_dir

INPUT_ROOT = "/home/gzu5140/TwINFER_KA/input_data"
PARAM_CSV = f"{INPUT_ROOT}/network_sweep/parameters.csv"

KADD_POS, KADD_NEG = 6.0, 0.8  # resolve_all_k_add sign-based defaults

# per-network (Hill n, k_add scale) chosen from a mean-field bifurcation + basin
# search: the point that gives >=2 well-separated stable fixed points with the
# best-balanced basins, favouring k_add x1 where possible (x2 drives protein
# counts to ~1e5 and slows the exact Gillespie a lot).
#   VSC  n=2  x1  -> 5 states (symmetric topology; splits ~57/43 from the empty start)
#   mCAD n=2  x1  -> 2 states
#   GSD  n=2  x2  -> 2 states, ~25/75 basins
#   HSC  n=4  x1  -> 3 states (needs steep Hill; k_add left at default for speed)
#   EMT  n=2  x2  -> 2 states, ~32/68 basins
NETWORKS = {
    "VSC":  dict(n_hill=2.0, kadd_scale=1.0, sim_time=6000, run_tag="multistate_6000steps"),
    "mCAD": dict(n_hill=2.0, kadd_scale=1.0, sim_time=2000, run_tag="multistate_2000steps"),
    "GSD":  dict(n_hill=2.0, kadd_scale=2.0, sim_time=2000, run_tag="multistate_2000steps"),
    "HSC":  dict(n_hill=4.0, kadd_scale=1.0, sim_time=2000, run_tag="multistate_2000steps"),
    "EMT":  dict(n_hill=2.0, kadd_scale=2.0, sim_time=2000, run_tag="multistate_2000steps"),
}
N_REPS = 3
PLAN = [(net, rep) for net in NETWORKS for rep in range(N_REPS)]


def build_matrices(M, n_hill, kadd_scale):
    """per-edge Hill exponent and k_add matrices (non-edges left at 0 / NaN)."""
    S = np.sign(M)
    n_genes = M.shape[0]
    n_matrix = np.zeros((n_genes, n_genes))
    k_add_matrix = np.full((n_genes, n_genes), np.nan)
    for i in range(n_genes):
        for j in range(n_genes):
            if S[i, j] != 0:
                n_matrix[i, j] = n_hill
                base = KADD_POS if S[i, j] > 0 else KADD_NEG
                k_add_matrix[i, j] = base * kadd_scale
    return n_matrix, k_add_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_index", type=int, default=0)
    args, _ = ap.parse_known_args()
    idx = args.config_index
    if not (0 <= idx < len(PLAN)):
        raise ValueError(f"config_index {idx} out of range 0..{len(PLAN) - 1}")

    net, rep = PLAN[idx]
    spec = NETWORKS[net]

    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    set_num_threads(cores)

    matrix_path = f"{INPUT_ROOT}/real_world_networks/{net}.txt"
    n_genes, M = read_input_matrix(matrix_path)
    n_matrix, k_add_matrix = build_matrices(M, spec["n_hill"], spec["kadd_scale"])

    out_dir = stage_dir(net, "simulate", run_tag=spec["run_tag"], make_latest=False)
    os.makedirs(f"{out_dir}/logs", exist_ok=True)

    print(f"[config {idx}] net={net} rep={rep} genes={n_genes} "
          f"n_hill={spec['n_hill']} kadd_scale={spec['kadd_scale']} "
          f"sim_time={spec['sim_time']} threads={get_num_threads()}", flush=True)
    print(f"  data_root={get_data_root()}  out_dir={out_dir}", flush=True)

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
        "type": f"{net}_multistate_rep_{rep}",
        "combinatorial_interaction_type": "additive",
        "n_matrix": n_matrix,
        "k_add_matrix": k_add_matrix,
        "use_csv_k_add": False,
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": cores,
    }

    label = "rows_" + "_".join(map(str, cfg["rows_to_use"][0]))
    path = process_param_set(cfg["rows_to_use"][0], label, cfg)
    print(f"[config {idx}] DONE -> {path}", flush=True)


if __name__ == "__main__":
    main()
