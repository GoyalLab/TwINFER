#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 16GB
#SBATCH -t 0:20:00
#SBATCH --job-name=celltag3_recon
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/celltag3_recon_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/celltag3_recon_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# "$PYTHON" /gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/175bdfed-a860-43a8-bebc-1b9019da7262/scratchpad/celltag3_recon.py   # [2026-09-30 repointed from an ephemeral Claude scratchpad to the rescued copy]
"$PYTHON" ${TWINFER_PROJECT_ROOT}/clean_code/benchmarks/network_benchmarks/score/formula_search/todo4v2_exploration/celltag3_recon.py
