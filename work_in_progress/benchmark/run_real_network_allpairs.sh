#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --job-name=real_networks_allpairs
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/real_networks_allpairs_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/real_networks_allpairs_%j.err
set -euo pipefail

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark

echo "[$(date)] real_networks ALL_PAIRS rerun (65 replicates, 7 networks)"
"$PYTHON" infer_real_network_allpairs.py --n-jobs 4 --n-cores-per-task 8
echo "[$(date)] Done"
