#!/bin/bash
# SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version).
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 12:00:00
#SBATCH --job-name=twsupp_test
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/twsupp_test_%j.out
#SBATCH --output=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/twsupp_test_%j.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/twsupp_test_%j.err
#SBATCH --error=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/twsupp_test_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

echo "[$(date)] TwinScore_supplement test: FM06 / correlation_high"
"$PYTHON" SUPERSEDED_apply_twinscore_supplement_fatemap.py FM06 --gene-set correlation_high --n-shuffles 2000 --n-cores 8
echo "[$(date)] Done"
