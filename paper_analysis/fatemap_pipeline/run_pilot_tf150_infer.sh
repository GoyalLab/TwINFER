#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 150GB
#SBATCH -t 24:00:00
#SBATCH --job-name=pilot150_infer
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/pilot150_infer_%j.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/pilot150_infer_%j.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/pilot150_infer_%j.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/pilot150_infer_%j.err
# PILOT (150 of the 581 expressed CollecTRI sources; gene set key tf_pilot150 in gene_sets_fm06.json): TwINFER inference, same settings as the 9 FM06 sets
# (A/B split, all pairs, 500 shuffles, 8 cores) so time and memory can be compared with detection_low (65 genes: 48 min, 11 GB).
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] PIPE=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline
PIPE=${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
cd $PIPE
echo "[$(date)] pilot150 inference start"
"$PYTHON" run_infer_fatemap_allpairs.py FM06 --gene-set tf_pilot150 --n-shuffles 500 --n-cores 8 --absplit
echo "[$(date)] Done"
