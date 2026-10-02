#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 4:00:00
#SBATCH --job-name=supp_phi_patch
#SBATCH --array=0-17
#SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/supp_patch_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/supp_patch_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

DATASETS=(FM06 FM08)
GENE_SETS=(variability_high variability_mid variability_low \
           detection_high detection_mid detection_low \
           correlation_high correlation_mid correlation_low)
N_GENE_SETS=${#GENE_SETS[@]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / N_GENE_SETS))]}
GENE_SET=${GENE_SETS[$((SLURM_ARRAY_TASK_ID % N_GENE_SETS))]}

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: $DATASET / $GENE_SET (phi-patched, A/B split, no gates)"
"$PYTHON" apply_twinscore_supplement_fatemap_allpairs.py "$DATASET" --gene-set "$GENE_SET" --n-shuffles 2000 --n-cores 8 --absplit
echo "[$(date)] Done"
