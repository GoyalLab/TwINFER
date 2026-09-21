#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --mem 64GB
#SBATCH -t 24:00:00
#SBATCH --job-name=multistate_sim
#SBATCH --array=0-14
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs/multistate_sim_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs/multistate_sim_%A_%a.err
set -euo pipefail

# IC-unset (empty start) sims for the real networks that CAN be pushed multistate,
# at per-network tuned (Hill n, k_add).  6000 cells, 50 threads.
#   array  0- 2 -> VSC  rep 0-2  (n=2, k_add x1 ; 6000 steps)
#   array  3- 5 -> mCAD rep 0-2  (n=2, k_add x1 ; 2000 steps)
#   array  6- 8 -> GSD  rep 0-2  (n=2, k_add x2 ; 2000 steps)
#   array  9-11 -> HSC  rep 0-2  (n=4, k_add x1 ; 2000 steps)
#   array 12-14 -> EMT  rep 0-2  (n=2, k_add x2 ; 2000 steps)
#
# conda hook is unreliable in batch -> call the env python directly.
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python
export PYTHONUNBUFFERED=1
export TWINFER_REPO_ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER

SCRIPT_DIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/heterogeneity_vs_regulation
mkdir -p /home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs
cd "$SCRIPT_DIR"

echo "[$(date)] job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
"$PY" -V
"$PY" -u real_network_multistate_sim.py --config_index "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done (exit $?)"
