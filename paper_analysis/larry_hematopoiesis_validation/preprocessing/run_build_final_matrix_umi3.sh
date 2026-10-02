#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=larry_umi3
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/umi3_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/umi3_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}

echo "[$(date)] Rebuilding final matrix with singletCode min_umi_cutoff=3 + criterion-4 rescue"
# [2026-09-30 commented out: pointed at a path that does not exist even in the original tree (code/build_final_matrix_umi3.py); the script sits next to this launcher]
# "$PYTHON" code/build_final_matrix_umi3.py
"$PYTHON" "${TWINFER_CODE_ROOT}/paper_analysis/larry_hematopoiesis_validation/preprocessing/build_final_matrix_umi3.py"
echo "[$(date)] Done"
