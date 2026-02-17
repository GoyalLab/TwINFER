# All packages needed to run TwINFER simulation and inference are listed here.
# If any of them are not installed, please install them using pip or conda env.
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
#%%
path_to_code_repo = "/home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER"
path_to_output = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/saturation_effects/"

#%%

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
# K_matrix = [[0, protein_1, 0,0,0], [0,0, protein_2,0,0], [0,0,0, protein_3,0], [0,0,0,0,protein_4], [0,0,0,0,0]]
base_configs = [
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/A_to_B/", #Path to the output folder
        "log_file": f"{path_to_output}/A_to_B.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "A_to_B_k_add_3",  # Name of the network used -- will be in the filename
        "use_given_K": True,
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [3]*2
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/A_to_B/", #Path to the output folder
        "log_file": f"{path_to_output}/A_to_B.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "A_to_B_k_add_6",  # Name of the network used -- will be in the filename
        "use_given_K": True,
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [6]*2
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*2], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/A_to_B/", #Path to the output folder
        "log_file": f"{path_to_output}/A_to_B.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "A_to_B_k_add_9",  # Name of the network used -- will be in the filename
        "use_given_K": True,
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [9]*2
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_and_B.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/A_and_B/", #Path to the output folder
        "log_file": f"{path_to_output}/A_and_B.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "A_and_B_k_add_3",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [3]*2
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_and_B.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/A_and_B/", #Path to the output folder
        "log_file": f"{path_to_output}/A_and_B.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "A_and_B_k_add_6",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [6]*2
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_and_B.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/A_and_B/", #Path to the output folder
        "log_file": f"{path_to_output}/A_and_B.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "A_and_B_kadd_9",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [9]*2
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_mutual_regulation.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/mutual_regulation/", #Path to the output folder
        "log_file": f"{path_to_output}/mutual_regulation.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "mutual_regulation_k_add_3",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [3]*3
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_mutual_regulation.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/mutual_regulation/", #Path to the output folder
        "log_file": f"{path_to_output}/mutual_regulation.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "mutual_regulation_k_add_6",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [6]*3
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_mutual_regulation.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/mutual_regulation/", #Path to the output folder
        "log_file": f"{path_to_output}/mutual_regulation.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "mutual_regulation_kadd_9",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [9]*3
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_feed_forward.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/feed_forward/", #Path to the output folder
        "log_file": f"{path_to_output}/feed_forward.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "feed_forward_k_add_3",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [3]*3
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_feed_forward.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/feed_forward/", #Path to the output folder
        "log_file": f"{path_to_output}/feed_forward.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "feed_forward_k_add_6",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [6]*3
    },
    {
        'n_cells': 6000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_feed_forward.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*3], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
        "output_folder": f"{path_to_output}/feed_forward/", #Path to the output folder
        "log_file": f"{path_to_output}/feed_forward.log",  # Name of the network used -- will be in the filename
        "log_pi_on": True,
        "type": "feed_forward_kadd_9",  # Name of the network used -- will be in the filename
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 48, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        "k_add_list": [9]*3
    },
]
# ==========================================================
# Select config by array index (with sanity check)
# ==========================================================
if not (0 <= config_index < len(base_configs)):
    config_index = config_index%len(base_configs)
    # raise ValueError(
    #     f"config_index={config_index} is out of range for {len(base_configs)} configs. "
    #     f"Set your SLURM array to 0–{len(base_configs) - 1}."
    # )
base_config = base_configs[config_index]
# Configure numba threads
set_num_threads(base_config["number_of_cores_per_parameter"])
print(f"🧠 Using {get_num_threads()} threads for config: {base_config['type']}")
# ==========================================================
# Import TwINFER gillespie script from local repo
# ==========================================================
import importlib
import sys
import os

# Make sure this path is on sys.path so Python can find the package
# Make sure this path is on sys.path so Python can find the package
if path_to_code_repo not in sys.path:
    sys.path.insert(0, path_to_code_repo)
# Now imports will be resolved relative to that repo
from TwINFER_function_scripts import gillespie_script_variations
importlib.reload(gillespie_script_variations)
from TwINFER_function_scripts.gillespie_script_variations import process_param_set

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
    cfg["type"] = f"{original_type}_{config_index}_{i}"
    print(f"▶️  Running replicate {i} with type={cfg['type']}")
    path_to_simulation_file = process_param_set(
        rows_to_use[0],
        labels[0],
        cfg
    )
    last_path = path_to_simulation_file
print(f"✅ Last simulation file saved as: {last_path}")
