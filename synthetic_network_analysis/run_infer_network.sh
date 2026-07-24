#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=62
#SBATCH --mem 30GB
#SBATCH -t 2:00:00
#SBATCH --output=/home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/synthetic_network_inference/logs/real_network_%A_%a.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/synthetic_network_inference/logs/real_network_%A_%a.err
set -eo pipefail


# ---- Conda activation that works both on Quest and inside Singularity
# if [ -f /projects/b1042/conda/etc/profile.d/conda.sh ]; then
#     source /projects/b1042/conda/etc/profile.d/conda.sh
#     conda activate /projects/b1042/conda/envs/twinfer-code
# elif [ -f /home/gzu5140/.conda/etc/profile.d/conda.sh ]; then
#     source /home/gzu5140/.conda/etc/profile.d/conda.sh
#     conda activate twinfer-code
# else
#     export PATH="/home/gzu5140/.conda/envs/twinfer-code/bin:$PATH"
#     echo "⚠️ Conda profile not found — using PATH-based activation"
# fi

# ---- Unbuffered output for live logging
export PYTHONUNBUFFERED=1

# echo "[$(date)] JOB $SLURM_JOB_ID / TASK $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Python: $(which python)"
python -V
echo

# ---- Run simulation directly from /home
cd "/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER/synthetic_network_analysis/"

echo "[$(date)] Starting analysis ..."
~/.conda/envs/twinfer-code/bin/python -u infer_network_simulation_cyclic.py
status=$?
echo "[$(date)] Simulation finished with exit code $status"
