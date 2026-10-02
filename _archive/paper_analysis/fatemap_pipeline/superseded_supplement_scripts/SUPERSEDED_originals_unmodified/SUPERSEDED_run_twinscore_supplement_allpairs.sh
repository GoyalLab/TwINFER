#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 4:00:00
#SBATCH --job-name=fm06_supp_allpairs
#SBATCH --array=0-1
#SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/fm06_supp_allpairs_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/fm06_supp_allpairs_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  echo "[$(date)] FM06 / correlation_high / TwinScore_supplement pooled (no gates/nan)"
  "$PYTHON" apply_twinscore_supplement_fatemap_allpairs.py FM06 --gene-set correlation_high --n-shuffles 2000 --n-cores 8
else
  echo "[$(date)] FM06 / correlation_high / TwinScore_supplement A/B split (no gates/nan)"
  "$PYTHON" apply_twinscore_supplement_fatemap_allpairs.py FM06 --gene-set correlation_high --n-shuffles 2000 --n-cores 8 --absplit
fi
echo "[$(date)] Done"
