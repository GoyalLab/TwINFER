#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --mem=10GB
#SBATCH --time=48:00:00
#SBATCH --job-name=A_to_B
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/slurm_log/slurmLog-%A-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/slurm_log/slurmLog-%A-%x.err
#SBATCH --array=0-39

eval "$(conda shell.bash hook)"
conda activate twinfer
start_index=$((600 * SLURM_ARRAY_TASK_ID))
end_index=$(start_index + 600)

path_to_parameter_sheet="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/simulation_details/parameters_3genes_positive_reg_pi_on_r_add_scaled.csv"
path_to_connectivity_matrix="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/simulation_details/connectivity_matrix_A_to_B.txt"
path_to_output_folder="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B/"
path_to_log_file="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/logs/A_to_B.jsonl"
type_of_interaction="A_to_B"

# Run Python script with matching CLI arguments
python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/TwINFER_function_scripts/gillespie_script.py \
    --path_to_connectivity_matrix "$path_to_connectivity_matrix" \
    --param_csv "$path_to_parameter_sheet" \
    --row_to_start "$start_index" \
    --row_to_end "$end_index" \
    --output_folder "$path_to_output_folder" \
    --log_file "$path_to_log_file" \
    --type "$type_of_interaction" \
    --number_parallel_processes 3 \
    --number_of_cores_per_parameter 10\
    --n_genes 2 \
    --n_cells 6000 \
    --simulation_time_before_division 1500 \
    --twin_simulation_time_after_division 48 \
    --twin_measurement_resolution 1