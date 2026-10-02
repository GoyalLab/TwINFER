"""
Gillespie simulations for every network in input_data/mixed_network_sweep/,
set up the same way as the network_sweep_final sims (same base config:
n_cells=6000, 6000 h pre-division, 48 h twin window, 1 h resolution,
additive combinatorial regulation, median parameter row [0]*n_genes per gene).

Layout
------
  * 48 networks x 3 simulation replicates = 144 simulations.
  * 6 SLURM array tasks; the 144 (network, replicate) jobs are split
    round-robin across tasks via --config_index (0..5), 24 jobs each.
  * Resumable: a (network, replicate) whose output CSV already exists in
    the output folder is skipped, so a timed-out array task can be resubmitted.

Output filenames match the network_sweep_final convention
(df_<net>_rep<k>_<timestamp>_ncells_6000_<net>_rep<k>_<hash>.csv).

Run one task locally:
    python mixed_network_sweep_sim.py --config_index 0
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]

import argparse
import glob
import os
import sys

from numba import set_num_threads, get_num_threads
from joblib import Parallel, delayed

import twinfer  # noqa: F401
from twinfer.simulation.gillespie_simulations import process_param_set, read_input_matrix

# ==========================================================
# Paths
# ==========================================================
PROJECT_ROOT = f'{TWINFER_PROJECT_ROOT}'
path_to_network_folder = f"{PROJECT_ROOT}/input_data/mixed_network_sweep/"
path_to_output_folder = f"{PROJECT_ROOT}/simulation_data/mixed_network_sweep/"
path_to_param_csv = f"{PROJECT_ROOT}/input_data/network_sweep/parameters.csv"

# ==========================================================
# Fixed job-layout parameters
# ==========================================================
n_array_jobs = 6
n_replicates = 3
n_parallel_per_job = 1

# numba threads per simulation: all cores SLURM gave this task
cores_per_job = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
cores_per_simulation = max(1, cores_per_job // n_parallel_per_job)

# ==========================================================
# Base simulation config (matches network_sweep_final)
# ==========================================================
BASE_CFG = {
    "n_cells": 6000,
    "simulation_time_before_division": 6000,
    "twin_simulation_time_after_division": 48,
    "twin_measurement_resolution": 1,
    "param_csv": path_to_param_csv,
    "output_folder": path_to_output_folder,
    "combinatorial_interaction_type": "additive",
    "log_pi_on": False,
}


def output_exists(net_name, rep):
    """True if a finished simulation CSV for this (network, replicate) is present."""
    label = f"{net_name}_rep{rep}"
    hits = glob.glob(os.path.join(path_to_output_folder, f"df_{label}_*_{label}_*.csv"))
    hits = [h for h in hits
            if not os.path.basename(h).startswith("simulation_before_division_df_")]
    return len(hits) > 0


def run_one_job(job):
    """Run a single simulation for one (network, replicate) combination."""
    set_num_threads(cores_per_simulation)

    net_name = job["network_name"]
    rep = job["replicate"]
    net_path = job["network_path"]
    label = f"{net_name}_rep{rep}"

    if output_exists(net_name, rep):
        print(f"  ⏭  {label} already done, skipping")
        return None

    n_genes, _ = read_input_matrix(net_path)
    rows_to_use = [0] * n_genes  # median parameter row, repeated per gene

    cfg = dict(BASE_CFG)
    cfg["path_to_connectivity_matrix"] = net_path
    cfg["log_file"] = f"{path_to_output_folder}/logs/{net_name}.jsonl"
    cfg["type"] = label

    print(f"  ▶️  {label} starting ({n_genes} genes, {cores_per_simulation} threads)")
    path_to_simulation_file = process_param_set(rows_to_use, label, cfg)
    print(f"  ✅ {label} done -> {path_to_simulation_file}")
    return path_to_simulation_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_index", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true",
                        help="list this task's jobs and exit without simulating")
    args, _ = parser.parse_known_args()
    config_index = args.config_index
    if not (0 <= config_index < n_array_jobs):
        raise ValueError(f"--config_index must be in 0..{n_array_jobs - 1}")

    print(f"Running array task #{config_index} of {n_array_jobs}")
    print(f"{cores_per_simulation} numba threads/simulation, "
          f"{n_parallel_per_job} concurrent simulation(s), {cores_per_job} total cores")

    os.makedirs(path_to_output_folder, exist_ok=True)
    os.makedirs(f"{path_to_output_folder}/logs", exist_ok=True)

    network_files = sorted(glob.glob(os.path.join(path_to_network_folder, "*.txt")))
    if not network_files:
        raise FileNotFoundError(f"No .txt network files found in {path_to_network_folder}")
    print(f"Found {len(network_files)} network file(s) in {path_to_network_folder}")

    all_jobs = []
    for net_path in network_files:
        net_name = os.path.splitext(os.path.basename(net_path))[0]
        for rep in range(n_replicates):
            all_jobs.append({"network_path": net_path,
                             "network_name": net_name,
                             "replicate": rep})
    print(f"Total jobs (networks x replicates): {len(all_jobs)} "
          f"({len(network_files)} networks x {n_replicates} replicates)")

    jobs_for_this_task = all_jobs[config_index::n_array_jobs]
    todo = [j for j in jobs_for_this_task
            if not output_exists(j["network_name"], j["replicate"])]
    print(f"▶️  Array task {config_index}: {len(jobs_for_this_task)} owned job(s), "
          f"{len(todo)} still to run")
    for j in jobs_for_this_task:
        done = output_exists(j["network_name"], j["replicate"])
        print(f"     {'done ' if done else 'TODO '} {j['network_name']}_rep{j['replicate']}")

    if args.dry_run:
        print("dry run — exiting before simulation")
        return

    results = Parallel(n_jobs=n_parallel_per_job, backend="multiprocessing", verbose=10)(
        delayed(run_one_job)(job) for job in jobs_for_this_task
    )

    print("✅ All jobs for this array task complete:")
    for r in results:
        print(f"   {r}")


if __name__ == "__main__":
    main()
