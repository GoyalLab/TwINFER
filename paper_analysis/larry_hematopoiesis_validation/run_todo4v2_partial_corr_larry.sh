#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=todo4v2_partial_corr_larry
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/todo4v2_partial_corr_larry_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/todo4v2_partial_corr_larry_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Scoring LARRY's 9 yscher panels with current_score (TODO4v2) + partial_corr_term (VSC
# indirect-correlation fix, see handoff/2026-09-17_vsc_and_e9pos0_followup.md). Some panels'
# z_scores_by_step.json are 600MB+ (correlation_mid/correlation_low), which killed this on the
# shared interactive node (SIGKILL, no error) -- submitting as a job instead, per the repo's
# "heavy compute goes through SLURM, not the login node" convention.

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/larry_hematopoiesis_validation"

echo "[$(date)] TODO4v2 + partial_corr_term LARRY yscher panel scoring"
"$PYTHON" apply_todo4v2_partial_corr_larry.py
echo "[$(date)] Done"
