#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 4:00:00
#SBATCH --job-name=hpsc_comp
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/hpsc_comp_%j.out
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/hpsc_comp_%j.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/hpsc_comp_%j.err
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/hpsc_comp_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

echo "[$(date)] HPSC_20260927 / chipatlas_perturbseq_panel (competitors: rho,ppcor,pidc,grnboost2 -- no genie3)"
"$PYTHON" run_competitors_fatemap.py HPSC_20260927 --gene-set chipatlas_perturbseq_panel \
    --methods rho,ppcor,pidc,grnboost2 --n-cores 8
echo "[$(date)] Done"
