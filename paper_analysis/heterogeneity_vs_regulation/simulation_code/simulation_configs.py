"""Settings of the six single-state / two-state simulation configurations of the heterogeneity_vs_regulation figure.

Figure  : heterogeneity_vs_regulation. Used by `simulate_heterogeneity_and_regulation.py`, which runs them.
Content : `get_base_configs()` returns the six configurations (a list of dictionaries, index = configuration number):
              0 A_B               no regulation, single state     parameter rows [0, 1]  (k_on 0.66)
              1 A_B_low_k_on      no regulation, low k_on         parameter rows [2, 2]  (k_on 0.12)
              2 A_B_high_k_on     no regulation, high k_on        parameter rows [3, 3]  (k_on 1.66)
              3 A_to_B            gene_1 -> gene_2, single state  parameter rows [0, 1]
              4 A_to_B_low_k_on   gene_1 -> gene_2, low k_on      parameter rows [2, 2]
              5 A_to_B_high_k_on  gene_1 -> gene_2, high k_on     parameter rows [3, 3]
          The two-state scenarios are formed later, in the analysis, by pairing a low_k_on and a high_k_on replicate.
          Common settings: 6,000 cells before division (= 6,000 twin pairs), 1,000 h burn-in, twins simulated for 48 h after division
          and sampled every hour, additive multiple-interaction model. Every key is explained in its comment.
Inputs  : the connectivity matrices and median_parameter.csv, found in `path_to_input` (the input_folder of the figure).
Usage   : from simulation_configs import get_base_configs
          configs = get_base_configs(path_to_input, path_to_output, num_cores_available)
"""


def get_base_configs(path_to_input, path_to_output, num_cores_available):
    """The six configurations; `path_to_output` is the folder that will hold one sub-folder per configuration type."""
    base_configs = [
        {
            'n_cells': 6000, #Number of cells before division (number of twin pairs)
            'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
            'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
            'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
            "path_to_connectivity_matrix": f"{path_to_input}/connectivity_matrix_2_gene_no_interaction.txt", #path to the connectivity matrix specifying the GRN to simulate
            "param_csv": f"{path_to_input}/median_parameter.csv", #Path to the parameters for all genes and interaction terms
            "rows_to_use": [[0,1]], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
            "output_folder": f"{path_to_output}/A_B/", #Path to the output folder
            "log_file": f"{path_to_output}/logs/A_B_median_parameter.log",  # Path to the log file for this simulation
            "type": "A_B",  # Name of the network used -- will be in the filename
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
            "number_of_cores_per_parameter": num_cores_available, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        },
        {
            'n_cells': 6000, #Number of cells before division (number of twin pairs)
            'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
            'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
            'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
            "path_to_connectivity_matrix": f"{path_to_input}/connectivity_matrix_2_gene_no_interaction.txt", #path to the connectivity matrix specifying the GRN to simulate
            "param_csv": f"{path_to_input}/median_parameter.csv", #Path to the parameters for all genes and interaction terms
            "rows_to_use": [[2]*2], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
            "output_folder": f"{path_to_output}/A_B_low_k_on/", #Path to the output folder
            "log_file": f"{path_to_output}/logs/A_B_median_parameter.log",  # Path to the log file for this simulation
            "type": "A_B_low_k_on",  # Name of the network used -- will be in the filename
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
            "number_of_cores_per_parameter": num_cores_available, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        },
        {
            'n_cells': 6000, #Number of cells before division (number of twin pairs)
            'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
            'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
            'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
            "path_to_connectivity_matrix": f"{path_to_input}/connectivity_matrix_2_gene_no_interaction.txt", #path to the connectivity matrix specifying the GRN to simulate
            "param_csv": f"{path_to_input}/median_parameter.csv", #Path to the parameters for all genes and interaction terms
            "rows_to_use": [[3]*2], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
            "output_folder": f"{path_to_output}/A_B_high_k_on", #Path to the output folder
            "log_file": f"{path_to_output}/logs/A_B_median_parameter.log",  # Path to the log file for this simulation
            "type": "A_B_high_k_on",  # Name of the network used -- will be in the filename
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
            "number_of_cores_per_parameter": num_cores_available, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        },
        {
            'n_cells': 6000, #Number of cells before division (number of twin pairs)
            'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
            'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
            'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
            "path_to_connectivity_matrix": f"{path_to_input}/connectivity_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
            "param_csv": f"{path_to_input}/median_parameter.csv", #Path to the parameters for all genes and interaction terms
            "rows_to_use": [[0,1]], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
            "output_folder": f"{path_to_output}/A_to_B/", #Path to the output folder
            "log_file": f"{path_to_output}/logs/A_to_B_median_parameter.log",  # Path to the log file for this simulation
            "type": "A_to_B",  # Name of the network used -- will be in the filename
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
            "number_of_cores_per_parameter": num_cores_available, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        },
        {
            'n_cells': 6000, #Number of cells before division (number of twin pairs)
            'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
            'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
            'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
            "path_to_connectivity_matrix": f"{path_to_input}/connectivity_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
            "param_csv": f"{path_to_input}/median_parameter.csv", #Path to the parameters for all genes and interaction terms
            "rows_to_use": [[2]*2], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
            "output_folder": f"{path_to_output}/A_to_B_low_k_on/", #Path to the output folder
            "log_file": f"{path_to_output}/logs/A_to_B_median_parameter.log",  # Path to the log file for this simulation
            "type": "A_to_B_low_k_on",  # Name of the network used -- will be in the filename
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
            "number_of_cores_per_parameter": num_cores_available, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        },
        {
            'n_cells': 6000, #Number of cells before division (number of twin pairs)
            'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
            'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
            'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
            "path_to_connectivity_matrix": f"{path_to_input}/connectivity_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
            "param_csv": f"{path_to_input}/median_parameter.csv", #Path to the parameters for all genes and interaction terms
            "rows_to_use": [[3]*2], #Rows in the parameter's csv file for each gene. Example - [0,0] will mean use row 0 parameters for both gene 1 and 2. The length should be equal to number of genes in the system. Ensure that each row in the parameter.csv has unique index.
            "output_folder": f"{path_to_output}/A_to_B_high_k_on/", #Path to the output folder
            "log_file": f"{path_to_output}/logs/A_to_B_median_parameter.log",  # Path to the log file for this simulation
            "type": "A_to_B_high_k_on",  # Name of the network used -- will be in the filename
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
            "number_of_cores_per_parameter": num_cores_available, #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
        }
    ]
    return base_configs
