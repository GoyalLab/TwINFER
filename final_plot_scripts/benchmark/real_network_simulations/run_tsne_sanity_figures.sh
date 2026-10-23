#!/bin/bash
#SBATCH -A p32655
#SBATCH -p short
#SBATCH -N 1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 04:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/tsne_sanity_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/tsne_sanity_%A_%a.err
set -eo pipefail
# task 0: all 12k cells at t=20 (plot_tsne_sisters_t20.py); task 1: 250 cells per time step x 48 (plot_tsne_spread_48tp_real_networks.py)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/final_plot_scripts/benchmark/real_network_simulations
SCRIPTS=(plot_tsne_sisters_t20.py plot_tsne_spread_48tp_real_networks.py)
/home/gzu5140/.conda/envs/twinfer-code/bin/python -u ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
