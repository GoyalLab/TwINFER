#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16GB
#SBATCH -t 00:45:00
#SBATCH --job-name=resume_infer
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference/logs/resume_infer_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference/logs/resume_infer_%j.err
set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis
~/.conda/envs/twinfer-code/bin/python -u resume_infer_real_networks_extra.py
