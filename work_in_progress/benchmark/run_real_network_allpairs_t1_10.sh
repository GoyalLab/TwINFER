#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --job-name=real_networks_allpairs_t1_10
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/real_networks_allpairs_t1_10_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/real_networks_allpairs_t1_10_%j.err
set -euo pipefail

# t1=10,t2=20 rerun of infer_real_network_allpairs.py (measure metrics farther from the division
# event). 65 replicates across 7 networks (same set as the original run); tested locally on one
# mCAD replicate (fast). Same resource shape as run_real_network_allpairs.sh.
# Output: analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs_t1_10/

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark

echo "[$(date)] real_networks t1=10 ALL_PAIRS rerun (65 replicates, 7 networks)"
"$PYTHON" infer_real_network_allpairs_t1_10.py --n-jobs 4 --n-cores-per-task 8
echo "[$(date)] Done"
