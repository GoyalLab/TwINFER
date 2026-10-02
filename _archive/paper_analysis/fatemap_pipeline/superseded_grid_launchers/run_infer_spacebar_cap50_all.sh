#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 12:00:00
#SBATCH --job-name=spacebar_cap50
#SBATCH --array=0-1
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/spacebar_cap50_%A_%a.out
#SBATCH --output=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/spacebar_cap50_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/spacebar_cap50_%A_%a.err
#SBATCH --error=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/spacebar_cap50_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

GENE_SETS=(correlation_high variability_high)
GENE_SET=${GENE_SETS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: SpaceBar / $GENE_SET / MAX_CLONE_SIZE=50"
"$PYTHON" run_infer_spacebar.py --gene-set "$GENE_SET" --n-shuffles 2000 --n-cores 8 --max-clone-size 50
echo "[$(date)] Done"
