#!/bin/bash
#SBATCH -A p32655
#SBATCH -p short
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem 64GB
#SBATCH -t 4:00:00
#SBATCH --job-name=infer_multistate
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate/logs/infer_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate/logs/infer_%j.err
set -eo pipefail

mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate/logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[$(date)] JOB $SLURM_JOB_ID on $(hostname)"
echo "Python: $(~/.conda/envs/twinfer-code/bin/python -V)"

cd "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis"

echo "[$(date)] Starting TwINFER inference over the multistate/seeded real-network sims ..."
~/.conda/envs/twinfer-code/bin/python -u infer_network_simulation_multistate.py
status=$?
echo "[$(date)] Inference finished with exit code $status"
exit $status
