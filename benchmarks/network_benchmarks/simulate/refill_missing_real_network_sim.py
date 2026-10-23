"""Refill missing real-network replicates (2026-10-18): GSD (needs 10, has 9) and Pluripotent (needs 10, has 6).
Same configs as the original runs, only the replicate label differs:
  GSD         -> real_network.py config (6000 pre-division steps, rows [0]*19), type GSD_0_<rep>, output simulation_data/real_data
  Pluripotent -> real_network_sim.py pattern (1000 pre-division steps, rows [0]*36), type Pluripotent_rep_<rep>,
                 output paper_analysis/Pluripotent/simulate/20260825_224511 (the folder holding reps 0-5)
    python refill_missing_real_network_sim.py --network Pluripotent --rep 6 [--smoke]
--smoke: 20 cells, 30 pre-division steps, output to --smoke_dir (checks the code path only).
"""
import argparse, os
from numba import set_num_threads
from twinfer.utils.paths import get_data_root as _g
from twinfer.simulation.gillespie_simulations import process_param_set

ROOT = _g().parent
ap = argparse.ArgumentParser()
ap.add_argument("--network", choices=["GSD", "Pluripotent"], required=True)
ap.add_argument("--rep", type=int, required=True)
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--smoke_dir", default=None)
a = ap.parse_args()
cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
if a.network == "GSD":
    n, steps, topo, out, typ = 19, 6000, "GSD.txt", f"{ROOT}/simulation_data/real_data", f"GSD_0_{a.rep}"
else:
    n, steps, topo, typ = 36, 1000, "Pluripotent.txt", f"Pluripotent_rep_{a.rep}"
    out = f"{ROOT}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511"
cfg = {
    "n_cells": 6000, "simulation_time_before_division": steps, "twin_simulation_time_after_division": 48,
    "twin_measurement_resolution": 1,
    "path_to_connectivity_matrix": f"{ROOT}/input_data/real_world_networks/{topo}",
    "param_csv": f"{ROOT}/input_data/network_sweep/parameters.csv",
    "output_folder": out, "log_file": f"{out}/logs/{a.network}.jsonl" if a.network == "GSD" else f"{out}/logs/Pluripotent.log",
    "type": typ, "number_of_parallel_parameters": 1, "number_of_cores_per_parameter": cores, "log_pi_on": False,
}
cfg["combinatorial_interaction_type" if a.network == "GSD" else "multiple_interaction_type"] = "additive"
if a.smoke:
    cfg.update(n_cells=20, simulation_time_before_division=30, output_folder=a.smoke_dir, log_file=f"{a.smoke_dir}/logs/x.log")
os.makedirs(cfg["output_folder"], exist_ok=True); os.makedirs(os.path.dirname(cfg["log_file"]), exist_ok=True)
set_num_threads(cores)
print(f"{a.network} rep {a.rep} type={typ} steps={cfg['simulation_time_before_division']} cores={cores} -> {cfg['output_folder']}", flush=True)
p = process_param_set([0] * n, "rows_" + "_".join(["0"] * n), cfg)
print("DONE", p)
