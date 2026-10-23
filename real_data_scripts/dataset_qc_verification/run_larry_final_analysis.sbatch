#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=larry_final
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/larry_final_%j.out
# LARRY: 5k-HVG PCA + clonal-vs-random pair distances on the UPDATED finalized_data/LARRY_data (Table 1 whitelist). Default paths: writes pca5k/ and pair_distance/ in
# analysis_data/dataset_qc_verification (the previous LARRY outputs are archived in archive_before_larry_table1_2026-10-16/).
set -euo pipefail
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
$PY pca5k.py LARRY
$PY pair_distance.py LARRY
