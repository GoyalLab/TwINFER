#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=larry_diptest_otsu
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/diptestotsu_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/diptestotsu_%j.err
set -euo pipefail

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /gpfs/projects/b1255/hzhang/TwINFER_KA

echo "[$(date)] Building dip-test + Otsu whitelist (real per-cell data, no external target)"
"$PYTHON" code/TwINFER/paper_analysis/larry_pipeline/whitelisting/build_diptest_otsu_whitelist.py
echo "[$(date)] Done"
