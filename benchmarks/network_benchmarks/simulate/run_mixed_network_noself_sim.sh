#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
##SBATCH --array=0-11   # [2026-10-17 replaced: 4 networks x 1 replicate]
##SBATCH --array=0-3   # [2026-10-17 replaced: replicates 1-2 added]
#SBATCH --array=0-7
#SBATCH --cpus-per-task=52
##SBATCH --mem 96GB   # [2026-10-17 replaced: 4 small sims]
#SBATCH --mem 16GB
##SBATCH -t 24:00:00   # [2026-10-17 replaced]
#SBATCH -t 06:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/simulation_data/mixed_network_noself/logs/slurm_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/simulation_data/mixed_network_noself/logs/slurm_%A_%a.err
set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# CONTROL EXPERIMENT: 4 autoregulated mixed networks (2 x n=6, 2 x n=10) with self-loops removed (input_data/mixed_network_noself/), 1 replicate each = 4 simulations,
# one per array task. Resumable: mixed_network_noself_sim.py skips any (network, replicate) whose output CSV already exists.

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
# LOG_DIR=/home/gzu5140/TwINFER_KA/simulation_data/mixed_network_noself/logs   # [2026-09-30 replaced by env.sh variable]
LOG_DIR=${TWINFER_PROJECT_ROOT}/simulation_data/mixed_network_noself/logs
mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

echo "Python: $(which python)"
python -V
echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID:-0} starting"

# ~/.conda/envs/twinfer-code/bin/python -u mixed_network_noself_sim.py \   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}" -u mixed_network_noself_sim.py \
    --config_index "$((${SLURM_ARRAY_TASK_ID:-0}))"
status=$?

echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID:-0} finished with exit code $status"
exit $status
