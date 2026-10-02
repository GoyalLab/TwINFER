from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# %%
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
import argparse
from numba import set_num_threads, get_num_threads
import importlib
import sys
import os
#%%
#Set output path


#Path to code repo
import twinfer
from twinfer.utils.paths import get_repo_root
from twinfer.simulation.gillespie_simulations import process_param_set
from twinfer.utils.paths import stage_dir


path_to_code_repo = get_repo_root()
#Path to output files
num_cores_available = 52
 #%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_index", type=int, default=0)
    args, _ = parser.parse_known_args()
    config_index = args.config_index
    print(f"🔧 Running base_config #{config_index}")

    # ==========================================================
    # Define base configurations (4 configs: indices 0..3, one per
    # {2-gene, 3-gene} x {positive, negative} autoregulation motif --
    # gene index 1 self-regulates (+1 or -1); all other edges and the
    # base parameter row are held fixed across the pair, isolating the
    # effect of autoregulation sign.
    # ==========================================================

    base_configs = [
        {
            'n_cells': 6000, #Number of cells before division (number of twin pairs)
            'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
            'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
            'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
            "path_to_connectivity_matrix": f"{TWINFER_PROJECT_ROOT}/input_data/autoreg_g2_pos.txt", #path to the connectivity matrix specifying the GRN to simulate
            "param_csv": f"{TWINFER_PROJECT_ROOT}/input_data/network_sweep/parameters.csv", #Path to the parameters for all genes and interaction terms
            "rows_to_use": [[0]*2], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
            "output_folder": f"{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g2_pos", #Path to the output folder
            "log_file": f"{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g2_pos/logs/autoreg_g2_pos.log",  # Path to the log file for this simulation
            "type": "autoreg_g2_pos",  # Name of the network used -- will be in the filename
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
            "number_of_cores_per_parameter": num_cores_available, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        },
        {
            'n_cells': 6000, #Number of cells before division (number of twin pairs)
            'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
            'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
            'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
            "path_to_connectivity_matrix": f"{TWINFER_PROJECT_ROOT}/input_data/autoreg_g2_neg.txt", #path to the connectivity matrix specifying the GRN to simulate
            "param_csv": f"{TWINFER_PROJECT_ROOT}/input_data/network_sweep/parameters.csv", #Path to the parameters for all genes and interaction terms
            "rows_to_use": [[0]*2], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
            "output_folder": f"{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g2_neg", #Path to the output folder
            "log_file": f"{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g2_neg/logs/autoreg_g2_neg.log",  # Path to the log file for this simulation
            "type": "autoreg_g2_neg",  # Name of the network used -- will be in the filename
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
            "number_of_cores_per_parameter": num_cores_available, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        },
        {
            'n_cells': 6000,
            'simulation_time_before_division': 1000,
            'twin_simulation_time_after_division': 48,
            'twin_measurement_resolution': 1,
            "path_to_connectivity_matrix": f"{TWINFER_PROJECT_ROOT}/input_data/autoreg_g3_pos.txt",
            "param_csv": f"{TWINFER_PROJECT_ROOT}/input_data/network_sweep/parameters.csv",
            "rows_to_use": [[0]*3],
            "output_folder": f"{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g3_pos",
            "log_file": f"{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g3_pos/logs/autoreg_g3_pos.log",
            "type": "autoreg_g3_pos",
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1,
            "number_of_cores_per_parameter": num_cores_available,
        },
        {
            'n_cells': 6000,
            'simulation_time_before_division': 1000,
            'twin_simulation_time_after_division': 48,
            'twin_measurement_resolution': 1,
            "path_to_connectivity_matrix": f"{TWINFER_PROJECT_ROOT}/input_data/autoreg_g3_neg.txt",
            "param_csv": f"{TWINFER_PROJECT_ROOT}/input_data/network_sweep/parameters.csv",
            "rows_to_use": [[0]*3],
            "output_folder": f"{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g3_neg",
            "log_file": f"{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g3_neg/logs/autoreg_g3_neg.log",
            "type": "autoreg_g3_neg",
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1,
            "number_of_cores_per_parameter": num_cores_available,
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

   # ==========================================================
    # Import TwINFER gillespie script from local repo
    # ==========================================================
    # Ensure output directory exists
    os.makedirs(base_config['output_folder'], exist_ok=True)

    # Prepare rows and labels
    rows_to_use = base_config['rows_to_use']   # e.g. [[7,7,7]]
    labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]

    original_type = base_config["type"]
    last_path = None

    # ==========================================================
    # Run 10 replicates of this config
    # ==========================================================
    for i in range(10):
        # shallow copy so we don't mutate base_config
        cfg = dict(base_config)
        cfg["type"] = f"{original_type}_rep_{i}"
        print(f"▶️  Running replicate {i} with type={cfg['type']}")
        path_to_simulation_file = process_param_set(
            rows_to_use[0],
            labels[0],
            cfg
        )
        last_path = path_to_simulation_file

    print(f"✅ Last simulation file saved as: {last_path}")
