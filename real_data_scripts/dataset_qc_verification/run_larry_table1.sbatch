#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=larry_t1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/larry_table1_%j.out
# LARRY stage 2 with the Weinreb et al. Table 1 whitelist (min UMIs: day2 500, day4 300, day6 300), corrected rescue, no Weinreb rule (i.e. as the finalized file),
# then the same annotation, then a comparison with the dip-test+Otsu run. New folder only: finalized_data/rebuild_fullsheet_2026-10-16/LARRY_table1_whitelist/
set -euo pipefail
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
NEW=/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/rebuild_fullsheet_2026-10-16
$PY rebuild_corrected_downstream.py larry_table1
( cd ../.. && SOURCE=$NEW/LARRY_table1_whitelist/filtered KNN=15 $PY paper_analysis/larry_hematopoiesis_validation/preprocessing/larry_raw_annotate.py )
$PY compare_larry_table1.py
