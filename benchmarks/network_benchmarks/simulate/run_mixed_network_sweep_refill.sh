#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=52
#SBATCH --mem 96GB
#SBATCH -t 12:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/simulation_data/mixed_network_sweep/logs/refill_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/simulation_data/mixed_network_sweep/logs/refill_%A_%a.err
set -eo pipefail
# Refill of the 28 mixed-sweep replicates lost when job 4056358 tasks 1-4 died with "Disk quota exceeded".
# 3 tasks; only not-yet-done (network, rep) pairs are split round-robin (MIXED_REFILL_BALANCED); resumable.
# Source code/TwINFER/env.sh before sbatch.
: "${TWINFER_CODE_ROOT:?source env.sh first}"
export MIXED_N_ARRAY_JOBS=3 MIXED_REFILL_BALANCED=1 PYTHONUNBUFFERED=1
cd "${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/simulate"
"${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}" -u mixed_network_sweep_sim.py --config_index "${SLURM_ARRAY_TASK_ID:-0}"
