#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=larry_check_excluded
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/check_excluded_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/check_excluded_%j.err
set -euo pipefail

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /gpfs/projects/b1255/hzhang/TwINFER_KA

echo "[$(date)] Checking excluded h5ad cells vs threshold"
"$PYTHON" code/TwINFER/paper_analysis/larry_pipeline/filtering/check_excluded_h5ad_cells.py
echo "[$(date)] Done"
