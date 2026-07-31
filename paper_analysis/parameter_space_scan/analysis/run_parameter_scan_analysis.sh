#!/bin/bash -l
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=42
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --job-name=Parameter-Analysis
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/logs/slurmLog-%A_%a-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/logs/slurmLog-%A_%a-%x.err

#Replace this with path to python in your conda environment while calling the python script: ~/.conda/envs/twinfer-code/bin/python

#These are examples of how to run the inference script analyze_parameter_scan_correlations.py for different classes of simulations in parameter scan

############################################################################################################
#Inferring regulation in a single-state using twins
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_rep_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_rep_B"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 1 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "single"

############################################################################################################
#Inferring regulation in a multi-state using twins
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_B_2_states"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair" \

############################################################################################################
#Inferring regulation in a single-state using random pairs of cells as twins (control) by setting remove_twin_structure to 1
# # Fixed input/output paths
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_and_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_and_B"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --remove_twin_structure 1

############################################################################################################
#Inferring regulation in a multi-state using random pairs of cells as twins (control) by setting remove_twin_structure to 1. 
# Using the same pairs as twins for generating two states esnures that the comparison is fair. But it is also possible to run without providing it.

path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_B_2_states"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair" \
  --csv "/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_1_20/A_B_2_states/two_state_pairs_final.csv" \
  --remove_twin_structure 1

#The commented out lines are for the remaining combinations that were analyzed in figure 2 and figure 3

path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_to_B_2_states"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair" \
  --csv "/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_1_20/A_to_B_2_states/two_state_pairs_final.csv" \
  --remove_twin_structure 1

path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_and_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_and_B_2_states"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair" \
  --csv "/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_1_20/A_and_B_2_states/two_state_pairs_final.csv" \
  --remove_twin_structure 1
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_rep_B/"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_rep_B_2_states"


# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair" \
  --csv "/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_1_20/A_rep_B_2_states/two_state_pairs_final.csv" \
  --remove_twin_structure 1

path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_rep_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_rep_B"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "single" \
  --remove_twin_structure 1


Fixed input/output paths
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_B"

Each array task processes a different chunk
Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025  \
  --mode "single" \
  --remove_twin_structure 1

  # Fixed input/output paths
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_to_B"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "single" \
  --remove_twin_structure 1

########
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_to_B_2_states"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair"

path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_B_2_states"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair" 


path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_and_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_and_B_2_states"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair"


path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_rep_B/"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_rep_B_2_states"


# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "pair" 


path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_rep_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_rep_B"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "single" 

path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_and_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_and_B"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "single" 



Fixed input/output paths
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_B"

Each array task processes a different chunk
Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025  \
  --mode "single" 

  # Fixed input/output paths
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_to_B"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --mode "single"