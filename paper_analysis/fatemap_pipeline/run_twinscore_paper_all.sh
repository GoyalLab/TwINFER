#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 3:00:00
#SBATCH --job-name=tp_paper
#SBATCH --array=0-44
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/tp_paper_%A_%a.out
#SBATCH --output=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/tp_paper_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/tp_paper_%A_%a.err
#SBATCH --error=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/tp_paper_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"
DATASETS=(FM06 FM01 Watermelon_naive Watermelon_lag Watermelon_late)
GENE_SETS=(variability_high variability_mid variability_low detection_high detection_mid detection_low correlation_high correlation_mid correlation_low)
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / 9))]}
GENE_SET=${GENE_SETS[$((SLURM_ARRAY_TASK_ID % 9))]}
echo "[$(date)] task $SLURM_ARRAY_TASK_ID: $DATASET / $GENE_SET (TwinScore-paper, merged A+B, Stage I-III, jackknife-50, BH q=0.05)"
"$PYTHON" -u twinscore_paper.py "$DATASET" "$GENE_SET" --n-jack 50
echo "[$(date)] task $SLURM_ARRAY_TASK_ID: competitors on the identical merged-A+B panel"
"$PYTHON" -u run_competitors_fatemap.py "$DATASET" --gene-set "$GENE_SET" --paper --methods rho,ppcor,pidc,genie3,grnboost2 --n-cores 4
echo "[$(date)] Done"
