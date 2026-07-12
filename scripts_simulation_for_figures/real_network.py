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
import importlib
import argparse
from numba import set_num_threads, get_num_threads

path_to_code_repo = "/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER"
path_to_output_folder = "/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/real_data/"

# ==========================================================
# Parse SLURM array index (0–4)
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config_index", type=int, default=0)
args, _ = parser.parse_known_args()
config_index = args.config_index
print(f"🔧 Running base_config #{config_index}")

base_configs = [
    # # ---------- CONFIG 0 ----------
     {
        'n_cells': 6000,
        'simulation_time_before_division': 6000,
        'twin_simulation_time_after_division': 48,
        'twin_measurement_resolution': 1,
        "path_to_connectivity_matrix": "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/real_world_networks/HSC.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"/home/gzu5140/Keerthana_b1042/TwINFER/input_data/real_world_networks/HSC_parameters.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*11],
        "output_folder": f"{path_to_output_folder}",
        "log_file": f"{path_to_output_folder}/logs/HSC.jsonl",
        "type": "HSC_balanced",
        "combinatorial_interaction_type": "additive",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": 56,
        "log_pi_on": False,
    },
    # {
    #     'n_cells': 6000,
    #     'simulation_time_before_division': 6000,
    #     'twin_simulation_time_after_division': 48,
    #     'twin_measurement_resolution': 1,
    #     "path_to_connectivity_matrix": "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/real_world_networks/mCAD.txt", #path to the connectivity matrix specifying the GRN to simulate
    #     "param_csv": f"/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
    #     "rows_to_use": [[0]*5],
    #     "output_folder": f"{path_to_output_folder}",
    #     "log_file": f"{path_to_output_folder}/logs/mCAD.jsonl",
    #     "type": "mCAD",
    #     "combinatorial_interaction_type": "additive",
    #     "number_of_parallel_parameters": 1,
    #     "number_of_cores_per_parameter": 56,
    #     "log_pi_on": False,
    # },
    # {
    #     'n_cells': 6000,
    #     'simulation_time_before_division': 6000,
    #     'twin_simulation_time_after_division': 48,
    #     'twin_measurement_resolution': 1,
    #     "path_to_connectivity_matrix": "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/real_world_networks/VSC.txt", #path to the connectivity matrix specifying the GRN to simulate
    #     "param_csv": f"/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
    #     "rows_to_use": [[0]*8],
    #     "output_folder": f"{path_to_output_folder}",
    #     "log_file": f"{path_to_output_folder}/logs/VSC.jsonl",
    #     "type": "VSC",
    #     "combinatorial_interaction_type": "additive",
    #     "number_of_parallel_parameters": 1,
    #     "number_of_cores_per_parameter": 56,
    #     "log_pi_on": False,
    # },
    # {
    #     'n_cells': 6000,
    #     'simulation_time_before_division': 6000,
    #     'twin_simulation_time_after_division': 48,
    #     'twin_measurement_resolution': 1,
    #     "path_to_connectivity_matrix": "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/real_world_networks/GSD.txt", #path to the connectivity matrix specifying the GRN to simulate
    #     "param_csv": f"/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
    #     "rows_to_use": [[0]*19],
    #     "output_folder": f"{path_to_output_folder}",
    #     "log_file": f"{path_to_output_folder}/logs/GSD.jsonl",
    #     "type": "GSD",
    #     "combinatorial_interaction_type": "additive",
    #     "number_of_parallel_parameters": 1,
    #     "number_of_cores_per_parameter": 56,
    #     "log_pi_on": False,
    # },
    ]
# ==========================================================
# Select config by array index (with sanity check)
# ==========================================================

config_index = config_index%len(base_configs)
if not (0 <= config_index < len(base_configs)):
    raise ValueError(
        f"config_index={config_index} is out of range for {len(base_configs)} configs. "
        f"Set your SLURM array to 0–{len(base_configs) - 1}."
    )

base_config = base_configs[config_index]

# Configure numba threads
set_num_threads(base_config["number_of_cores_per_parameter"])
print(f"🧠 Using {get_num_threads()} threads for config: {base_config['type']}")

#  ==========================================================
# Import TwINFER gillespie script from local repo
# ==========================================================
# Make sure this path is on sys.path so Python can find the package
if path_to_code_repo not in sys.path:
    sys.path.insert(0, path_to_code_repo)

# Now imports will be resolved relative to that repo
from TwINFER_function_scripts import gillespie_script_variations
importlib.reload(gillespie_script_variations)
from TwINFER_function_scripts.gillespie_script_variations import process_param_set
# from TwINFER_function_scripts import pause_run_sim
# importlib.reload(pause_run_sim)
# from TwINFER_function_scripts.pause_run_sim import process_param_set

# Ensure output directory exists
os.makedirs(base_config['output_folder'], exist_ok=True)

# Prepare rows and labels
rows_to_use = base_config['rows_to_use']   # e.g. [[7,7,7]]
labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]

original_type = base_config["type"]
last_path = None

# ==========================================================
# Run 20 replicates of this config
# ==========================================================
for i in range(20):
    # shallow copy so we don't mutate base_config
    cfg = dict(base_config)
    cfg["type"] = f"{original_type}_{config_index}_{i}"

    print(f"▶️  Running replicate {i} with type={cfg['type']}")
    path_to_simulation_file = process_param_set(
        rows_to_use[0],
        labels[0],
        cfg
    )
    last_path = path_to_simulation_file
print(f"✅ Last simulation file saved as: {last_path}")
