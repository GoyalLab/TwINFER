#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --mem=6GB
#SBATCH --time=4:00:00
#SBATCH --job-name=drift_simulation
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/drift_simulations/logs/slurmLog-%A_%a-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/drift_simulations/logs/slurmLog-%A_%a-%x.err

eval "$(conda shell.bash hook)"
conda activate twinfer-code
echo 1
~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/drift_multiple_state/simulate_drift_multiple_states.py