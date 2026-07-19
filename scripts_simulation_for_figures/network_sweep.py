import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numba
import tqdm
import scipy
import seaborn
import os
import sys
import glob
import importlib
import argparse
from numba import set_num_threads, get_num_threads
from joblib import Parallel, delayed

path_to_code_repo = "/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER"
path_to_network_folder = "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep_final/"
path_to_output_folder = "/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/network_sweep_final/"
path_to_param_csv = f"/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep/parameters.csv"

# ==========================================================
# Fixed job-layout parameters (hardcoded, matching original style)
# ==========================================================
n_array_jobs = 1
n_replicates = 5
n_parallel_per_job = 1
cores_per_job = 58
cores_per_simulation = cores_per_job // n_parallel_per_job  # 60 // 3 = 20

# ==========================================================
# Parse SLURM array index (0–9)
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config_index", type=int, default=0)
args, _ = parser.parse_known_args()
config_index = args.config_index
print(f"Running array task #{config_index} of {n_array_jobs}")
print(f"{cores_per_simulation} numba threads/simulation, "
      f"{n_parallel_per_job} concurrent simulations, {cores_per_job} total cores")

# ==========================================================
# Import TwINFER gillespie script from local repo
# ==========================================================
if path_to_code_repo not in sys.path:
    sys.path.insert(0, path_to_code_repo)

from TwINFER_function_scripts import gillespie_script_variations
importlib.reload(gillespie_script_variations)
from TwINFER_function_scripts.gillespie_script_variations import process_param_set, read_input_matrix

os.makedirs(path_to_output_folder, exist_ok=True)
os.makedirs(f"{path_to_output_folder}/logs", exist_ok=True)

# ==========================================================
# Discover every network file in the folder
# ==========================================================
network_files = sorted(glob.glob(os.path.join(path_to_network_folder, "*.txt")))
if not network_files:
    raise FileNotFoundError(f"No .txt network files found in {path_to_network_folder}")
print(f"Found {len(network_files)} network file(s) in {path_to_network_folder}")

# ==========================================================
# Build the full job list: every (network, replicate) pair
# ==========================================================
all_jobs = []
required = ['grn_n6_e5_pos100_density_rep0', 'grn_n6_e5_pos100_density_rep2',
        'grn_n6_e17_pos100_density_rep0', 
        'grn_n6_e17_pos100_density_rep1', 'grn_n6_e17_pos100_density_rep2']
for net_path in network_files:
    net_name = os.path.splitext(os.path.basename(net_path))[0]
    if net_name not in required:
        continue
    for rep in range(n_replicates):
        all_jobs.append({"network_path": net_path, "network_name": net_name, "replicate": rep})

print(f"Total jobs (networks x replicates): {len(all_jobs)} "
      f"({len(network_files)} networks x {n_replicates} replicates)")

# ==========================================================
# Partition jobs round-robin across SLURM array tasks
# ==========================================================
jobs_for_this_task = all_jobs[config_index::n_array_jobs]
print(f"▶️  This array task is running {len(jobs_for_this_task)} job(s): "
      f"{[j['network_name'] + '_rep' + str(j['replicate']) for j in jobs_for_this_task]}")

# ==========================================================
# Worker: runs ONE (network, replicate) job
# ==========================================================
def run_one_job(job):
    """
    Runs a single simulation for one network/replicate combination.

    Args:
        job (dict): {'network_path', 'network_name', 'replicate'}.

    Returns:
        str: Path to the saved simulation output CSV.
    """
    set_num_threads(cores_per_simulation)

    net_name = job["network_name"]
    rep = job["replicate"]
    net_path = job["network_path"]

    n_genes, _ = read_input_matrix(net_path)
    rows_to_use = [0] * n_genes  # median parameter row, repeated per gene
    label = f"{net_name}_rep{rep}"

    cfg = {
        'n_cells': 6000,
        'simulation_time_before_division': 6000,
        'twin_simulation_time_after_division': 48,
        'twin_measurement_resolution': 1,
        "path_to_connectivity_matrix": net_path,
        "param_csv": path_to_param_csv,
        "output_folder": path_to_output_folder,
        "log_file": f"{path_to_output_folder}/logs/{net_name}.jsonl",
        "type": label,
        "combinatorial_interaction_type": "additive",
        "number_of_cores_per_parameter": cores_per_simulation,
        "log_pi_on": False,
    }

    print(f"  ▶️  {label} starting ({n_genes} genes, {cores_per_simulation} threads)")
    path_to_simulation_file = process_param_set(
        rows_to_use,
        label,
        cfg
    )
    print(f"  ✅ {label} done -> {path_to_simulation_file}")
    return path_to_simulation_file

# ==========================================================
# Run this array task's jobs, n_parallel_per_job at a time
# ==========================================================
results = Parallel(n_jobs=n_parallel_per_job, backend="multiprocessing", verbose=10)(
    delayed(run_one_job)(job) for job in jobs_for_this_task
)

print("✅ All jobs for this array task complete:")
for r in results:
    print(f"   {r}")