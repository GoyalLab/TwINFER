#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 8:00:00
#SBATCH --job-name=tsv2_merged
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/tsv2_merged_%A_%a.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/tsv2_merged_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/tsv2_merged_%A_%a.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/tsv2_merged_%A_%a.err
# New TwinScore (indicator product x s(z_reg)) on FM06 with A and B merged into one sample. usage: sbatch [--array=0-8] run_twinscore_v2_merged_fm06.sh [gene_set [extra args]]
# with --array the gene set is taken from the array index (variability_high ... correlation_low, same order as the other FM06 arrays).
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
GENE_SETS=(variability_high variability_mid variability_low detection_high detection_mid detection_low correlation_high correlation_mid correlation_low)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then GS=${GENE_SETS[$SLURM_ARRAY_TASK_ID]}; EXTRA="${1:-}"; else GS=${1:-correlation_high}; EXTRA="${2:-}"; fi
# PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] PIPE=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline
PIPE=${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
cd $PIPE
echo "[$(date)] $GS: twinscore v2 merged  $EXTRA"
"$PYTHON" twinscore_v2_merged_fm06.py "$GS" --n-null 200 --n-shuffles 500 --n-cores 8 $EXTRA
echo "[$(date)] Done"
