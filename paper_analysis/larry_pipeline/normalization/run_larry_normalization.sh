#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 1:00:00
#SBATCH --job-name=larry_norm
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/norm_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/norm_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Re-runs only larry_normalization_comparison.ipynb (the QC'd matrix from
# larry_preprocessing.ipynb is unchanged and already cached on disk) after
# fixing the H_PRIME/scale_K bug for log1pCP10k and log1pPF.

JUPYTER=/home/gzu5140/.conda/envs/twinfer-code/bin/jupyter
# NBDIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_analysis   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] NBDIR=${TWINFER_PROJECT_ROOT}/code/TwINFER/real_data_analysis
NBDIR=${TWINFER_CODE_ROOT}/notebooks/real_data_analysis

echo "[$(date)] Executing larry_normalization_comparison.ipynb"
"$JUPYTER" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=twinfer-code \
    --ExecutePreprocessor.timeout=1800 \
    "$NBDIR/larry_normalization_comparison.ipynb"

echo "[$(date)] Done"
