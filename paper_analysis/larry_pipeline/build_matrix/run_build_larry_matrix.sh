#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --job-name=larry_matrix
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/build_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/build_%j.err
set -euo pipefail

# Builds the combined LARRY cell x gene matrix + lineage-barcode assignment.
# See /home/gzu5140/.claude/plans/can-you-run-the-wiggly-wall.md for the plan
# and code/build_larry_matrix.py for the implementation. Run on a compute node
# via sbatch rather than the login shell: the login session's memcg killed an
# earlier interactive attempt at ~2.7GB RSS.

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /gpfs/projects/b1255/hzhang/TwINFER_KA

echo "[$(date)] Starting LARRY matrix build"
"$PYTHON" code/TwINFER/paper_analysis/larry_pipeline/build_matrix/build_larry_matrix.py
echo "[$(date)] Done"
