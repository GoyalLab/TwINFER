#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 64GB
#SBATCH -t 12:00:00
#SBATCH --job-name=gs_score
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/gs_score_%j.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/gs_score_%j.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/gs_score_%j.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/gs_score_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
GS=${1:?gene set key}
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"
# TwinScore(phi): final gated-bootstrap TwinScore_supplement (A/B split, python PIDC as D), parallel bootstrap; needs the z_dagger json of run_gs_infer.sh.
# Written next to the other gene sets in twinscore_supp_gated_bootstrap/ (new file names only; nothing existing is touched).
echo "[$(date)] $GS: TwinScore(phi) scoring"
# --out-dir /projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/twinscore_supp_gated_bootstrap   # [2026-09-30 replaced by env.sh variable]
"$PYTHON" apply_twinscore_supplement_fatemap_gated_bootstrap_parallel.py FM06 --gene-set "$GS" --n-shuffles 2000 --n-cores 16 --absplit \
    --out-dir ${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/twinscore_supp_gated_bootstrap
echo "[$(date)] Done"
