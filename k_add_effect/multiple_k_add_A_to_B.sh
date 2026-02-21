#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --mem=10GB
#SBATCH --time=4:00:00
#SBATCH --job-name=A_to_B
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/k_add_simulations/slurm_log/slurmLog-%A-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/k_add_simulations/slurm_log/slurmLog-%A-%x.err
#SBATCH --array=0-9

eval "$(conda shell.bash hook)"
conda activate twinfer-code
start_index=$((200 * SLURM_ARRAY_TASK_ID))
end_index=$((start_index+2000))

path_to_code_repo="/home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/"
path_to_connectivity_matrix="${path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_to_B.txt"

#Path to parameter file generated using parameters_for_multiple_k_add.ipynb
path_to_parameter="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/test/k_add_simulations/effect_of_k_add_sampling_positive_with_reps.csv"
path_to_output_folder="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/test/"
path_to_log_file="${path_to_output_folder}/logs/A_to_B.jsonl"
type_of_interaction="A_to_B"

# Run Python script with matching CLI arguments
python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/TwINFER_function_scripts/gillespie_script_variations.py \
    --path_to_connectivity_matrix "$path_to_connectivity_matrix" \
    --param_csv "$path_to_parameter" \
    --row_to_start "$start_index" \
    --row_to_end "$end_index" \
    --output_folder "$path_to_output_folder" \
    --log_file "$path_to_log_file" \
    --type "$type_of_interaction" \
    --number_parallel_processes 1 \
    --number_of_cores_per_parameter 10\
    --n_genes 2 \
    --n_cells 6000 \
    --simulation_time_before_division 1000 \
    --twin_simulation_time_after_division 48 \
    --twin_measurement_resolution 1