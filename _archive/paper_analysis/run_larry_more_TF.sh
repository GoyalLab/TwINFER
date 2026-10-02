#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=55
#SBATCH --mem 150GB
#SBATCH -t 24:00:00
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/analysisData/Top_50_most_variable_genes_12032026//%A_%a.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/analysisData/Top_50_most_variable_genes_12032026/%A_%a.err
set -eo pipefail


# ---- Conda activation that works both on Quest and inside Singularity
if [ -f /projects/b1042/conda/etc/profile.d/conda.sh ]; then
    source /projects/b1042/conda/etc/profile.d/conda.sh
    conda activate /projects/b1042/conda/envs/twinfer
elif [ -f /home/gzu5140/.conda/etc/profile.d/conda.sh ]; then
    source /home/gzu5140/.conda/etc/profile.d/conda.sh
    conda activate twinfer
else
    export PATH="/home/gzu5140/.conda/envs/twinfer-code/bin:$PATH"
    echo "⚠️ Conda profile not found — using PATH-based activation"
fi

# ---- Unbuffered output for live logging
export PYTHONUNBUFFERED=1

# echo "[$(date)] JOB $SLURM_JOB_ID / TASK $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Python: $(which python)"
python -V
echo

# ---- Run simulation directly from /home
cd "//home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/"

echo "[$(date)] Starting analysis ..."
~/.conda/envs/twinfer-code/bin/python -u run_larry_more_TF.py
status=$?
echo "[$(date)] Simulation finished with exit code $status"
