#!/bin/bash
# ===== SUPERSEDED 2026-09-08 by run_infer_allpairs20_all9.sh =====
# (consolidated: 9 gene sets as one array 0-8, 20 cores / 16h / 5000 null draws).
# Original body kept commented below per repo convention.
#
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem 16GB
#SBATCH -t 8:00:00
#SBATCH --job-name=infer_ap20_correlation
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/infer_ap20_correlation_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/infer_ap20_correlation_%j.err
# set -euo pipefail
# 
# ALL_PAIRS rerun on the rebuilt (build_panel_set.py-method) gene sets, 20 cores / 16GB / 8h.
# One job per criterion (3 gene sets each). No OUT_SUFFIX -> canonical {GENE_SET}_t2_t4_allpairs/.
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA
# 
# for GENE_SET in correlation_high correlation_mid correlation_low; do
#     echo "[$(date)] TwINFER ALL_PAIRS run on $GENE_SET (t1=2, t2=4), 20 cores"
#         /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_infer_correlation_high.py
#     echo "[$(date)] Done ($GENE_SET)"
# done
#     ALL_PAIRS=1 GENE_SET="$GENE_SET" N_CORES=20 "$PYTHON" \
