#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=52
#SBATCH --mem 96GB
#SBATCH -t 12:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/simulation_data/real_data/logs/refill_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/simulation_data/real_data/logs/refill_%A_%a.err
set -eo pipefail
# Tasks 0-3: Pluripotent reps 6-9 (have 0-5). Task 4: GSD rep 3 (GSD_0_0..2 x3 files exist = 9; this adds the 10th).
# Source code/TwINFER/env.sh before sbatch.
: "${TWINFER_CODE_ROOT:?source env.sh first}"
export PYTHONUNBUFFERED=1
cd "${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/simulate"
PY="${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}"
T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -le 3 ]; then NET=Pluripotent; REP=$((6+T)); else NET=GSD; REP=3; fi
"$PY" -u refill_missing_real_network_sim.py --network "$NET" --rep "$REP"
