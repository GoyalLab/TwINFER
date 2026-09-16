#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=larry_spring_whitelist_v2
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/spring_whitelist_v2_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/spring_whitelist_v2_%j.err
set -euo pipefail

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /gpfs/projects/b1255/hzhang/TwINFER_KA

echo "[$(date)] Building SPRING-style whitelist v2 (tuned to target cell count)"
"$PYTHON" code/TwINFER/paper_analysis/larry_pipeline/whitelisting/build_spring_whitelist_v2.py
echo "[$(date)] Done"
