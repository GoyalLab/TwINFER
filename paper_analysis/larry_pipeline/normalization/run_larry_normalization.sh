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

# Re-runs only larry_normalization_comparison.ipynb (the QC'd matrix from
# larry_preprocessing.ipynb is unchanged and already cached on disk) after
# fixing the H_PRIME/scale_K bug for log1pCP10k and log1pPF.

JUPYTER=/home/gzu5140/.conda/envs/twinfer-code/bin/jupyter
NBDIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_analysis

echo "[$(date)] Executing larry_normalization_comparison.ipynb"
"$JUPYTER" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=twinfer-code \
    --ExecutePreprocessor.timeout=1800 \
    "$NBDIR/larry_normalization_comparison.ipynb"

echo "[$(date)] Done"
