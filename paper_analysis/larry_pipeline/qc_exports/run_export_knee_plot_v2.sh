#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=larry_knee_v2
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/kneev2_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/kneev2_%j.err
set -euo pipefail

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /gpfs/projects/b1255/hzhang/TwINFER_KA

echo "[$(date)] Exporting updated knee-plot data (v2 threshold + h5ad boundary cells)"
"$PYTHON" code/TwINFER/paper_analysis/larry_pipeline/qc_exports/export_knee_plot_data_v2.py
echo "[$(date)] Done"
