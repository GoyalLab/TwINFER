#Script to generate multiple states before division and this is inherited by both daughter cells
#All packages needed to run TwINFER simulation and inference are listed here. 

#%%
#If any of them are not installed, please install them using pip or conda env.
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
import joblib
from itertools import product

#%%
path_to_code_repo = "/home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/"
output_folder = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/del/"

#Common path to data files
base_config_list = [{
    'n_cells': 6000, #Number of cells before division (number of twin pairs)
    'simulation_time_before_division': 1500, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
    'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
    'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
    "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
    "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_param.csv", #Path to the parameters for all genes and interaction terms
    "rows_to_use": [[0,0]], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system.
    "output_folder": f"{output_folder}", #Path to the output folder
    "log_file": f"{output_folder}/log.jsonl", #Path to the log file
    "type": "A_to_B_2_states",  # Name of the network used -- will be in the filename
    "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
    "number_of_cores_per_parameter": 18, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
},
{
    'n_cells': 6000, #Number of cells before division (number of twin pairs)
    'simulation_time_before_division': 1500, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
    'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
    'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
    "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_B.txt", #path to the connectivity matrix specifying the GRN to simulate
    "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_param.csv", #Path to the parameters for all genes and interaction terms
    "rows_to_use": [[0,0]], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system.
    "output_folder": f"{output_folder}", #Path to the output folder
    "log_file": f"{output_folder}/log.jsonl", #Path to the log file
    "type": "A_B_no_reg_2_states",  # Name of the network used -- will be in the filename
    "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
    "number_of_cores_per_parameter": 18, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
}]
base_config = base_config_list[1] #change this to one to simulate the regulation case

#%%
#Path to TwINFER code repository
import os, sys
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import set_num_threads, get_num_threads

# sys.path.append(str(path_to_code_repo))
set_num_threads(base_config['number_of_cores_per_parameter'])
print("Threads Numba will use:", get_num_threads())

import importlib
import gillespie_script_drift
importlib.reload(gillespie_script_drift)
from gillespie_script_drift import process_param_set
#%%
for i in np.arange(1, 20):
    os.makedirs(base_config['output_folder'], exist_ok=True)
    rows_to_use = base_config['rows_to_use']
    labels = [f"rows_{'_'.join(map(str, row))}_{i}" for row in rows_to_use]
    path_to_simulation_file = process_param_set(rows_to_use[0], labels[0], base_config)
    print(f"Saved the simulation file as {path_to_simulation_file}")

#%%
# for i in np.arange(0, 20):
#     os.makedirs(base_config['output_folder'], exist_ok=True)
#     rows_to_use = base_config['rows_to_use']
#     labels = [f"rows_{'_'.join(map(str, row))}_{i}" for row in rows_to_use]
#     path_to_simulation_file = process_param_set(rows_to_use[0], labels[0], base_config)
#     print(f"Saved the simulation file as {path_to_simulation_file}")

# import numpy as np, matplotlib.pyplot as plt

# def scale_up(t):  return 1 + (2-1)*0.5*(1+np.tanh((t-10)/5))
# def scale_down(t):return 1/scale_up(t)

# t = np.linspace(0,40,400)
# plt.plot(t, scale_up(t), label="Increasing k_on(t)")
# plt.plot(t, scale_down(t), label="Decreasing k_on(t)")
# plt.axvline(10, color="gray", linestyle="--", label="Division time (t=10h)")
# plt.xlabel("Time (hours)"); plt.ylabel("Multiplier")
# plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
