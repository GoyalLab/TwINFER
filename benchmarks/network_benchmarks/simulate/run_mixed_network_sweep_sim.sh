#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --array=0-5
#SBATCH --cpus-per-task=52
#SBATCH --mem 96GB
#SBATCH -t 24:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/simulation_data/mixed_network_sweep/logs/slurm_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/simulation_data/mixed_network_sweep/logs/slurm_%A_%a.err
set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# 48 networks x 3 replicates = 144 simulations, split round-robin across 6
# array tasks (24 jobs each). Resumable: mixed_network_sweep_sim.py skips any
# (network, replicate) whose output CSV already exists, so a timed-out task
# can be resubmitted safely.

# ---- Conda activation (Quest / Singularity)
if [ -f /projects/b1042/conda/etc/profile.d/conda.sh ]; then
    source /projects/b1042/conda/etc/profile.d/conda.sh
    conda activate /projects/b1042/conda/envs/twinfer-code
elif [ -f /home/gzu5140/.conda/etc/profile.d/conda.sh ]; then
    source /home/gzu5140/.conda/etc/profile.d/conda.sh
    conda activate twinfer-code
else
    export PATH="/home/gzu5140/.conda/envs/twinfer-code/bin:$PATH"
    echo "⚠️ Conda profile not found — using PATH-based activation"
fi

export PYTHONUNBUFFERED=1

# SCRIPT_DIR=/home/gzu5140/TwINFER_KA/code/TwINFER/synthetic_network_analysis   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR="${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/simulate"
# LOG_DIR=/home/gzu5140/TwINFER_KA/simulation_data/mixed_network_sweep/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/simulation_data/mixed_network_sweep/logs
mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "Python: $(which python)"
python -V
echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID:-0} starting"

# ~/.conda/envs/twinfer-code/bin/python -u mixed_network_sweep_sim.py \   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}" -u mixed_network_sweep_sim.py \
    --config_index "$((${SLURM_ARRAY_TASK_ID:-0}))"
status=$?

echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID:-0} finished with exit code $status"
exit $status
