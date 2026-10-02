#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 6:00:00
#SBATCH --job-name=fm06_supp_gb
#SBATCH --array=0-8
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/fm06_supp_gb_%A_%a.out
#SBATCH --output=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/fm06_supp_gb_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/fm06_supp_gb_%A_%a.err
#SBATCH --error=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/fm06_supp_gb_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

GENE_SETS=(variability_high variability_mid variability_low \
           detection_high detection_mid detection_low \
           correlation_high correlation_mid correlation_low)
GENE_SET=${GENE_SETS[$SLURM_ARRAY_TASK_ID]}

# FINAL version: bootstrapped noise floors everywhere null_sd was used + h-gate on phi (median placeholder), no phi clip.
# Reads z_dagger_fm06_${GENE_SET}_absplit_allpairs.json (alpha=0.999 inference); writes to
# analysis_data/fm06/data/twinscore_supp_gated_bootstrap/ (own folder; nothing existing is overwritten).
echo "[$(date)] task $SLURM_ARRAY_TASK_ID: FM06 / $GENE_SET / TwinScore_supplement, A-B split, gated bootstrap (FINAL)"
"$PYTHON" apply_twinscore_supplement_fatemap_gated_bootstrap.py FM06 --gene-set "$GENE_SET" --n-shuffles 2000 --n-cores 8 --absplit
echo "[$(date)] Done"
