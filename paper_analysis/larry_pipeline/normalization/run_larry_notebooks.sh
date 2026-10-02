#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 1:00:00
#SBATCH --job-name=larry_notebooks
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/notebooks_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/notebooks_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Executes larry_preprocessing.ipynb then larry_normalization_comparison.ipynb
# (which depends on the first's output) on a compute node -- the login
# session's memcg previously killed a pandas load of just one raw count
# matrix at ~2.7GB RSS, and these notebooks load the full 95k x 25k combined
# sparse matrix (~1.4GB mtx on disk).

JUPYTER=/home/gzu5140/.conda/envs/twinfer-code/bin/jupyter
# NBDIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_analysis   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] NBDIR=${TWINFER_PROJECT_ROOT}/code/TwINFER/real_data_analysis
NBDIR=${TWINFER_CODE_ROOT}/notebooks/real_data_analysis

echo "[$(date)] Executing larry_preprocessing.ipynb"
"$JUPYTER" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=twinfer-code \
    --ExecutePreprocessor.timeout=1800 \
    "$NBDIR/larry_preprocessing.ipynb"

echo "[$(date)] Executing larry_normalization_comparison.ipynb"
"$JUPYTER" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=twinfer-code \
    --ExecutePreprocessor.timeout=1800 \
    "$NBDIR/larry_normalization_comparison.ipynb"

echo "[$(date)] Done"
