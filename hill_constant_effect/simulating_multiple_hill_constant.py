# %% 
# This is the script to simulate different values of Hill constant as a function of mean estimated protein level. The default is 1.

# # Code to simulate a synthetic GRN and infer the network using TwINFER
# 
path_to_data = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/hill_constant_effect/"
path_to_code_repo = ""
# %% [markdown]
# ## Details about the simulation

### Set this for both running the simulation
base_config = {
    'n_cells': 6000, #Number of cells before division (number of twin pairs)
    'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
    'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
    'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
    "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
    "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_param.csv", #Path to the parameters for all genes and interaction terms
    "rows_to_use": [[0,0]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
    "output_folder": f"{path_to_data}/simulations/", #Path to folder to store simulation 
    "log_file": f"{path_to_data}/logs/multiple_hill_constant_median_parameters.jsonl", #Path to the log file
    "type": "A_to_B",  # Name of the network used -- will be in the filename
    "number_parallel_processes": 3, #Number of parameters to be run in parallel
    "number_of_cores_per_parameter": 9, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
    "scale_k": [[0,1],[0,0]]
}

# %%
import sys
from pathlib import Path
sys.path.append(str(path_to_code_repo))

import copy
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import os
import numpy as np
from numba import set_num_threads, get_num_threads
set_num_threads(base_config['number_of_cores_per_parameter'])
print("Threads Numba will use:", get_num_threads())

import importlib
from TwINFER_function_scripts import gillespie_script_variations
importlib.reload(gillespie_script_variations)
from TwINFER_function_scripts.gillespie_script_variations import process_param_set

# %%
# Calculation functions
import importlib
from TwINFER_function_scripts import correlation_analysis_functions
from TwINFER_function_scripts import correlation_analysis_helpers

importlib.reload(correlation_analysis_functions)
importlib.reload(correlation_analysis_helpers)

from TwINFER_function_scripts.correlation_analysis_functions import (
    calculate_pairwise_gene_gene_correlation_matrix,
    check_system_in_steady_state,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_pair_correlations,
    differentiate_single_state_reg_and_multiple_states,
    identify_reg_if_multiple_states,
    get_cross_correlations
)

# Helper functions
from TwINFER_function_scripts.correlation_analysis_helpers import (
    extract_param_index,
    read_input_matrix,
    get_param_data, 
    plot_matrix_as_heatmap,
    print_summary,
    plot_network
)

# %% [markdown]
# ## Simulate the gene expression in a population of cells
# 
# The code simulates gene expression based on a GRN (described by the interaction matrix) and expression of each gene is defined by parameters (each row in the parameter sheet) using the Gillespie algorithm.
# %%
import numpy as np
from tqdm.auto import tqdm
import sys
import copy
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("chunk_id",nargs="?", default =0,type=int, help="Which chunk of the job array to run (0–4)")
args = parser.parse_args()
chunk_id = args.chunk_id
scale_k_pattern = np.array([
    [0, 1],
    [0, 0]
])
# --- setup ---
scale_k_values = np.linspace(0.1, 4, 10)
all_tasks = list(product(scale_k_values, range(20))) #can be reduced to 5/10 replicates

# split into 5 chunks
n_splits = 5
chunk_size = int(np.ceil(len(all_tasks) / n_splits))
chunks = [all_tasks[i*chunk_size:(i+1)*chunk_size] for i in range(n_splits)]

tasks_to_run = chunks[chunk_id]

# --- loop ---
tasks = []
for scale_val, rep in tasks_to_run:
    run_config = copy.deepcopy(base_config)

    scaled_matrix = scale_k_pattern * scale_val
    run_config["scale_k"] = scaled_matrix

    scale_str = f"{scale_val:.1f}"
    run_config["type"] = f"{run_config['type']}_scale_k_{scale_str}"

    os.makedirs(run_config['output_folder'], exist_ok=True)

    rows_to_use = run_config['rows_to_use']
    labels = [
        f"rows_{'_'.join(map(str, row))}_scale_k_{scale_str}_rep_{rep}"
        for row in rows_to_use
    ]

    param_sets = list(zip(rows_to_use, labels))
    for rows, label in param_sets:
        tasks.append((rows, label, copy.deepcopy(run_config)))

results = Parallel(n_jobs=base_config['number_parallel_processes'])(
    delayed(process_param_set)(rows, label, run_config)
    for rows, label, run_config in tqdm(tasks, desc=f"Chunk {chunk_id} all jobs")
)