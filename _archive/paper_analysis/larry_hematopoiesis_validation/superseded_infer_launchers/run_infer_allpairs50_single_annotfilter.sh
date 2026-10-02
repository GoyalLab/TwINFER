#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --mem 64GB
#SBATCH -t 10:00:00
#SBATCH --job-name=infer_allpairs50_annotfilter_single
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/infer_allpairs50_annotfilter_single_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/infer_allpairs50_annotfilter_single_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}

: "${GENE_SET:?GENE_SET must be set, e.g. sbatch --export=ALL,GENE_SET=correlation_mid run_infer_allpairs50_single_annotfilter.sh}"

echo "[$(date)] TwINFER ALL_PAIRS run on $GENE_SET (t1=2, t2=4), 50 cores, annotation-filtered twins, single-gene-set job (10hr budget)"
# /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_infer_correlation_high.py   # [2026-09-30 replaced by env.sh variable]
ALL_PAIRS=1 OUT_SUFFIX=_50core_annotfilter GENE_SET="$GENE_SET" N_CORES=50 INPUT_DIR=twinfer_input_annotfilter "$PYTHON" \
    ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_infer_correlation_high.py
echo "[$(date)] Done ($GENE_SET)"
