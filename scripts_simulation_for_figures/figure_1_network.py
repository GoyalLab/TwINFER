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

path_to_code_repo = "/home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER"
path_to_output_folder = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/figure_4/"
# ==========================================================
# Parse SLURM array index (0–4)
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config_index", type=int, default=0)
args, _ = parser.parse_known_args()
config_index = args.config_index
print(f"🔧 Running base_config #{config_index}")

# ==========================================================
# Define base configurations (5 configs: indices 0..4)
# ==========================================================
protein_1_2 = 125265.1379158951
protein_1 = 29965.104
protein_2_ffl = 119763.6917
protein_2_cascade = 119763.6917
protein_3_cascade = 122672.08966666667
mutual_reg_2_3 = 125265.1379158951
K_matrix = [
    [0, protein_1, protein_1, 0, 0,0,0,0,0,0,0,0,0,0],
    [0, 0, 0, protein_2_ffl, 0,0,0,0,0,0,0,0,0,0],
    [0]*14,
    [0]*14,
    [0, 0, 0, 0, 0, protein_1, protein_1, 0, 0,0,0,0,0,0],
    [0,0,0,0,0,0,protein_2_ffl,protein_2_ffl,0,0,0,0,0,0],
    [0]*14,
    [0]*14,
    [0]*14,
    [0,0,0,0,0,0,0,0,0,0,protein_1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0, 0, protein_2_cascade, 0, 0],
    [0,0,0,0,0,0,0,0,0,0, 0,0, protein_3_cascade,protein_3_cascade],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,mutual_reg_2_3],
    [0,0,0,0,0,0,0,0,0,0,0,0,mutual_reg_2_3,0]
]

base_configs = [
    # # ---------- CONFIG 0 ----------
     {
        'n_cells': 6000,
        'simulation_time_before_division': 6000,
        'twin_simulation_time_after_division': 48,
        'twin_measurement_resolution': 1,
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_figure_1_network.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        "output_folder": f"{path_to_output_folder}",
        "log_file": f"{path_to_output_folder}/figure_1_network.jsonl",
        "type": "figure_1_network",
        "use_given_K": True,
        "K_to_use":K_matrix,
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": 56,
        "log_pi_on": False,
    }
    ]
# ==========================================================
# Select config by array index (with sanity check)
# ==========================================================
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

# Ensure output directory exists
os.makedirs(base_config['output_folder'], exist_ok=True)

# Prepare rows and labels
rows_to_use = base_config['rows_to_use']   # e.g. [[7,7,7]]
labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]

original_type = base_config["type"]
last_path = None

# ==========================================================
# Run 2 replicates of this config
# ==========================================================
for i in range(2):
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
