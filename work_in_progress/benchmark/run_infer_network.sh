#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=62
#SBATCH --mem 64GB
#SBATCH -t 12:00:00
#SBATCH --job-name=infer_real_networks
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference/logs/infer_network_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference/logs/infer_network_%j.err
set -eo pipefail

mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference/logs

# ---- Unbuffered output for live logging
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[$(date)] JOB $SLURM_JOB_ID on $(hostname)"
echo "Python: $(~/.conda/envs/twinfer-code/bin/python -V)"
echo

# ---- Full TwINFER inference over every real-network simulation replicate
# (GSD/HSC/mCAD/VSC restricted to BEELINE's replicate set; B_cell_activation/
# Circadian_cycle/EMT/Pluripotent run over every replicate on disk). See the
# module docstring in infer_network_simulation_real_network.py for details.
cd "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis"

echo "[$(date)] Starting full real-network inference ..."
~/.conda/envs/twinfer-code/bin/python -u infer_network_simulation_real_network.py
status=$?
echo "[$(date)] Inference finished with exit code $status"
exit $status
