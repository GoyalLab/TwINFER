#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=9
#SBATCH --mem 32GB
#SBATCH -t 12:00:00
#SBATCH --job-name=hpsc_top100_infer
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/hpsc_top100_infer_%j.out
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/hpsc_top100_infer_%j.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/hpsc_top100_infer_%j.err
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/hpsc_top100_infer_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
export OPENBLAS_NUM_THREADS=8
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"
echo "[$(date)] full infer_with_twinfer, unfiltered 100-gene panel (40 TF + 60 target), n_shuffles=2000"
"$PYTHON" run_infer_hpsc_endoderm_top100.py --n-shuffles 2000 --n-cores 8
echo "[$(date)] Done"
