#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 64GB
#SBATCH -t 12:00:00
#SBATCH --job-name=hs054_infer
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/hs054_infer_%j.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/hs054_infer_%j.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/hs054_infer_%j.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/hs054_infer_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"
GENE_SETS="variability_high variability_mid variability_low detection_high detection_mid detection_low correlation_high correlation_mid correlation_low"
for GS in $GENE_SETS; do
  echo "[$(date)] $GS: HS054_Sot48h single-timepoint inference"
  "$PYTHON" run_infer_hs054.py --gene-set "$GS" --n-shuffles 2000 --n-cores 8
done
echo "[$(date)] Done"
