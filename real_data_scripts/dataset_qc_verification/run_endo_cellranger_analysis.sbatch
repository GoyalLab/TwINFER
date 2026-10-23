#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=endo_cr
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/endo_cr_%j.out
# hPSC_endo_yscher: 5k-HVG PCA + pair distances on the CELL RANGER counts (endo_counts_mito20_min200genes.h5ad) instead of the kb-python X.
# Default paths: rewrites pca5k/ and pair_distance/ for this dataset (previous outputs archived in archive_before_endo_cellranger_2026-10-16/).
set -euo pipefail
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
$PY pca5k.py hPSC_endo_yscher
$PY pair_distance.py hPSC_endo_yscher
