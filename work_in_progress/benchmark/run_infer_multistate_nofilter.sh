#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem 16GB
#SBATCH -t 4:00:00
#SBATCH --job-name=infer_multistate_nofilter
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate_nofilter/logs/infer_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate_nofilter/logs/infer_%j.err
set -eo pipefail

mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate_nofilter/logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[$(date)] JOB $SLURM_JOB_ID on $(hostname)"
echo "Python: $(~/.conda/envs/twinfer-code/bin/python -V)"

cd "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis"

# ---- TwINFER inference over the multistate/seeded real-network sims, with
# the Step 1-3 coverage gates disabled (same rationale as
# run_infer_network_nofilter.sh). Memory sized from job 5501498's actual
# usage (~6GB of 64GB requested for the filtered equivalent); no network here
# is larger than GSD (n=19, 342 pairs), well within what HSC (n=11) already
# validated locally.
echo "[$(date)] Starting TwINFER inference over the multistate/seeded real-network sims (no-filter) ..."
~/.conda/envs/twinfer-code/bin/python -u infer_network_simulation_multistate_nofilter.py
status=$?
echo "[$(date)] Inference finished with exit code $status"
exit $status
