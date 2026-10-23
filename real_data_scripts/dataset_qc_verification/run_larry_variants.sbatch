#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=larry_opts
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/larry_variants_%j.out
# LARRY comparison: option 2 (Weinreb rules ii+iii) and option 3 (one copy per duplicated cell barcode + clone) built from the CURRENT file into finalized_data/rebuild_fullsheet_2026-10-16/variants/,
# each with its own 5k-HVG PCA and pair distances (QCV_FIN / QCV_OUT redirect), then compared with the current file. finalized_data/LARRY_data is not modified.
set -euo pipefail
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
for v in opt2 opt3; do
  $PY larry_variants.py $v
  export QCV_FIN=/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/rebuild_fullsheet_2026-10-16/variants/$v QCV_OUT=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/larry_variants/$v
  mkdir -p $QCV_OUT
  $PY pca5k.py LARRY
  $PY pair_distance.py LARRY
  unset QCV_FIN QCV_OUT
done
$PY compare_larry_variants.py
