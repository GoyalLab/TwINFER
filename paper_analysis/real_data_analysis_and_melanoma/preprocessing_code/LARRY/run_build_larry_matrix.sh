#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --job-name=larry_matrix
# Figure 5 preprocessing, LARRY stage 1: the 15 inDrops libraries of the authors + their barcode FASTQ -> combined cell x gene matrix and the UMI table of the lineage barcodes
# (>= 10 reads per (library, cell, UMI, barcode), barcodes merged within Hamming distance 3). Input: raw_data/LARRY_authors_zips_and_barcode_fastq.
# Output: <REBUILD>/LARRY/processed (override with LARRY_PROCESSED). Run on a compute node: the login node kills it (~2.7 GB).
# Usage: sbatch run_build_larry_matrix.sh
set -euo pipefail
ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
FIG=$ROOT/final_figure_data/real_data_analysis_and_melanoma
RAW=${FIG5_RAW_DATA:-$FIG/raw_data}                                  # links to the raw data (see raw_data/)
FINAL=$FIG/final_data                                                # links to the final objects
REBUILD=${FIG5_REBUILD_DIR:-$FIG/analysis_data/preprocessing_rebuild}  # rebuilt files go here, never into final_data
CODE=$ROOT/code/TwINFER/paper_analysis/real_data_analysis_and_melanoma/preprocessing_code
PY=${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}   # conda activate does not work in batch jobs: call the environment python directly
LOGS=$REBUILD/logs; mkdir -p "$LOGS"
export TWINFER_PROJECT_ROOT=$ROOT
echo "[$(date)] Starting LARRY matrix build"
$PY "$CODE/LARRY/build_larry_matrix.py"
echo "[$(date)] Done"
