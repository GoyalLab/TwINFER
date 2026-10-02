#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --job-name=tsspec
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/tsspec_%A_%a.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/tsspec_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/tsspec_%A_%a.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/tsspec_%A_%a.err
# TwinScore pair procedure (spec of 2026-09-25) on FM06, ANALYTIC null only, scored against CollecTRI.
# usage: sbatch --array=0-1 run_twinscore_spec_fm06.sh [gene_set [nulls [n_null]]]   nulls: comma list of analytic (uncorrected z_reg), zreg_corrected (within-clone-shuffle centre + sd for z_reg), clone, bootstrap     array 0 = merged (A+B one sample), 1 = ab (A=t1, B=t2)
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
MODES=(merged ab)
MODE=${MODES[${SLURM_ARRAY_TASK_ID:-0}]}
GS=${1:-correlation_high}
NULLS=${2:-analytic}
NNULL=${3:-200}
# PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] PIPE=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline
PIPE=${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
export OMP_NUM_THREADS=4
cd $PIPE
mkdir -p logs
echo "[$(date)] $GS mode=$MODE nulls=$NULLS n_null=$NNULL"
"$PYTHON" twinscore_spec_fm06.py "$GS" --mode "$MODE" --nulls "$NULLS" --n-null "$NNULL"
echo "[$(date)] Done"
