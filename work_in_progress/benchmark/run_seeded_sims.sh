#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --mem 64GB
#SBATCH -t 24:00:00
#SBATCH --job-name=seeded_sim
#SBATCH --array=0-14
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs/seeded_sim_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs/seeded_sim_%A_%a.err
set -euo pipefail

# Seeded-IC sims: 6000 cells split evenly across each network's k stable basins,
# so the population is BALANCED across states. Same per-network (n, k_add) as
# run_multistate_sims.sh.
#   array  0- 2 -> VSC  rep 0-2   |   3- 5 -> mCAD   |   6- 8 -> GSD
#   array  9-11 -> HSC  rep 0-2   |  12-14 -> EMT
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python
export PYTHONUNBUFFERED=1
export TWINFER_REPO_ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER

SCRIPT_DIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/heterogeneity_vs_regulation
mkdir -p /home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs
cd "$SCRIPT_DIR"

echo "[$(date)] job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
"$PY" -V
"$PY" -u real_network_seeded_sim.py --config_index "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done (exit $?)"
