#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --mem=10GB
#SBATCH --time=48:00:00
#SBATCH --job-name=A_rep_B
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/k_add_simulations/slurm_log/slurmLog-%A-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/k_add_simulations/slurm_log/slurmLog-%A-%x.err
#SBATCH --array=0-9

eval "$(conda shell.bash hook)"
conda activate twinfer
start_index=$((200 * SLURM_ARRAY_TASK_ID))
end_index=$((start_index+350))

path_to_parameter="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/k_add_simulations/simulation_details/effect_of_k_add_sampling_repression_with_reps.csv"
path_to_connectivity_matrix="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/k_add_simulations/simulation_details/connectivity_matrix_A_rep_B.txt"
path_to_output_folder="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/k_add_simulations/A_rep_B_with_reps/"
path_to_log_file="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/k_add_simulations/logs/A_rep_B.jsonl"
type_of_interaction="A_rep_B"

# Run Python script with matching CLI arguments
python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/TwINFER_function_scripts/gillespie_script.py \
    --path_to_connectivity_matrix "$path_to_connectivity_matrix" \
    --param_csv "$path_to_parameter" \
    --row_to_start "$start_index" \
    --row_to_end "$end_index" \
    --output_folder "$path_to_output_folder" \
    --log_file "$path_to_log_file" \
    --type "$type_of_interaction" \
    --number_parallel_processes 3 \
    --number_of_cores_per_parameter 10\
    --n_genes 2 \
    --n_cells 6000 \
    --simulation_time_before_division 1000 \
    --twin_simulation_time_after_division 48 \
    --twin_measurement_resolution 1