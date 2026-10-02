#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 6:00:00
#SBATCH --job-name=sb_cab_infer_supp
#SBATCH --array=0-2
#SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/sb_cab_infer_supp_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/sb_cab_infer_supp_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

GENE_SETS=(correlation_high correlation_mid correlation_low)
GS=${GENE_SETS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: SpaceBar centroid A/B / $GS -- infer_with_twinfer (no gates)"
"$PYTHON" run_infer_spacebar_centroid_ab.py --gene-set "$GS" --n-shuffles 500 --n-cores 8
echo "[$(date)] TwinScore_supplement scoring"
"$PYTHON" apply_twinscore_supplement_spacebar_centroid_ab.py --gene-set "$GS" --n-shuffles 2000 --n-cores 8
echo "[$(date)] Done"
