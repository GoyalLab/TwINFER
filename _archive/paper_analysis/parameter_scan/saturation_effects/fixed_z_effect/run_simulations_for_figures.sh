#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --mem 10GB
#SBATCH -t 24:00:00
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/fixed_z/%A_%a.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/fixed_z/%A_%a.err
#SBATCH --array=0-11
set -eo pipefail


# ---- Conda activation that works both on Quest and inside Singularity
if [ -f /projects/b1042/conda/etc/profile.d/conda.sh ]; then
    source /projects/b1042/conda/etc/profile.d/conda.sh
    conda activate /projects/b1042/conda/envs/twinfer
elif [ -f /home/gzu5140/.conda/etc/profile.d/conda.sh ]; then
    source /home/gzu5140/.conda/etc/profile.d/conda.sh
    conda activate twinfer
else
    export PATH="/home/gzu5140/.conda/envs/twinfer/bin:$PATH"
    echo "⚠️ Conda profile not found — using PATH-based activation"
fi

# ---- Unbuffered output for live logging
export PYTHONUNBUFFERED=1

# echo "[$(date)] JOB $SLURM_JOB_ID / TASK $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Python: $(which python)"
python -V
echo

# ---- Run simulation directly from /home
cd "/home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/saturation_effects/fixed_z_effect/"

echo "[$(date)] Starting simulation ..."
~/.conda/envs/twinfer-code/bin/python -u fixed_z_simulations.py --config_index $((${SLURM_ARRAY_TASK_ID:-0}))
status=$?
echo "[$(date)] Simulation finished with exit code $status"

echo "[$(date)] DONE — job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"
