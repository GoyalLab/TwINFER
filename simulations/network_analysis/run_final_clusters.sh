#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 20GB
#SBATCH -t 2:00:00
#SBATCH --job-name=final_clusters
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs/final_clusters_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs/final_clusters_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# 8 networks -> 8 worker processes (one CSV each, single-threaded per worker).
# Peak ~1 GB/worker, dominated by the initial 2-column read of the 36M-row files.
export N_WORKERS=8
# PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python   # [2026-09-30 replaced by env.sh variable]
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
# cd /projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}
echo "[$(date)] starting on $(hostname)"
$PY -u code/plot_real_network_final_clusters.py
echo "[$(date)] done"
