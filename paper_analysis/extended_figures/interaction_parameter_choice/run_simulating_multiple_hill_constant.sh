#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --mem=10GB
#SBATCH --time=12:00:00
#SBATCH --job-name=multiple-hill-constant
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/logs/slurmLog-%A_%a-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/logs/slurmLog-%A_%a-%x.err
#SBATCH --array=0-4   # run 5 chunks (0,1,2,3,4)

eval "$(conda shell.bash hook)"
conda activate twinfer

python simulating_multiple_hill_constant.py $SLURM_ARRAY_TASK_ID
