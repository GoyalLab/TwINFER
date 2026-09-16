#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=62
#SBATCH --mem 24GB
#SBATCH -t 12:00:00
#SBATCH --job-name=infer_real_networks_nofilter
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter/logs/infer_network_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter/logs/infer_network_%j.err
set -eo pipefail

mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter/logs

# ---- Unbuffered output for live logging
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[$(date)] JOB $SLURM_JOB_ID on $(hostname)"
echo "Python: $(~/.conda/envs/twinfer-code/bin/python -V)"
echo

# ---- Full TwINFER inference over every ORIGINAL (pre-multistate-work)
# real-network simulation replicate, with the Step 1-3 coverage gates
# disabled so ranked_edges/twinScore reaches ~every directed pair instead of
# only the 3-19% that used to clear the default alpha=0.01 correlation
# screen. Memory sized from actual usage of the equivalent filtered runs
# (jobs 5234749, 5501498: ~6-7GB used of 64GB requested) plus margin for
# Pluripotent (n=36, 1260 possible pairs, previously only ~3% ever reached
# the heavier Stage 2-4 machinery -- untested at full no-filter load). See
# infer_network_simulation_real_network_nofilter.py's module docstring for
# the exact parameter changes.
cd "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis"

echo "[$(date)] Starting full real-network inference (no-filter) ..."
~/.conda/envs/twinfer-code/bin/python -u infer_network_simulation_real_network_nofilter.py
status=$?
echo "[$(date)] Inference finished with exit code $status"
exit $status
