#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --mem=10GB
#SBATCH --time=04:00:00
#SBATCH --job-name=multiple-partition
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/binomial_partition/%a-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/binomial_partition/slurmLog-%A_%a-%x.err

eval "$(conda shell.bash hook)"
conda activate twinfer-code

python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/binomial_partitioning/simulating_binomial_partition.py
