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
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference/logs

# ---- Unbuffered output for live logging
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[$(date)] JOB $SLURM_JOB_ID on $(hostname)"
# echo "Python: $(~/.conda/envs/twinfer-code/bin/python -V)"   # [2026-09-30 replaced by env.sh variable]
echo "Python: $("${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}" -V)"
echo

# ---- Full TwINFER inference over every real-network simulation replicate
# (GSD/HSC/mCAD/VSC restricted to BEELINE's replicate set; B_cell_activation/
# Circadian_cycle/EMT/Pluripotent run over every replicate on disk). See the
# module docstring in infer_network_simulation_real_network.py for details.
# cd "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis"   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/infer"

echo "[$(date)] Starting full real-network inference ..."
# ~/.conda/envs/twinfer-code/bin/python -u infer_network_simulation_real_network.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}" -u infer_network_simulation_real_network.py
status=$?
echo "[$(date)] Inference finished with exit code $status"
exit $status
