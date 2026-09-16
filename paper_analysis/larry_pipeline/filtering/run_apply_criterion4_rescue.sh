#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=larry_criterion4
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/crit4_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/crit4_%j.err
set -euo pipefail

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /gpfs/projects/b1255/hzhang/TwINFER_KA

echo "[$(date)] Applying criterion-4 (cross-sample barcode-combination) rescue on top of production whitelist"
"$PYTHON" code/TwINFER/paper_analysis/larry_pipeline/filtering/apply_criterion4_rescue_pipeline.py
echo "[$(date)] Done"
