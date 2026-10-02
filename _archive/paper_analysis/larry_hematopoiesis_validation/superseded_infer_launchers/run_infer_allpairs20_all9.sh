#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem 16GB
#SBATCH -t 16:00:00
#SBATCH --job-name=infer_ap20
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/infer_ap20_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/infer_ap20_%A_%a.err
#SBATCH --array=0-8
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# ALL_PAIRS 5000-shuffle run on the CURRENT gene sets (var reverted to top-HVG terciles, corr
# ranked by median |rho|), reading the log1p-CP10k NORMALIZED panels (twinfer_input_cp10k).
# 9 gene sets, one job each (array 0-8), 20 cores / 16GB / 16h. Driver pins minimum_null_draws=5000.
# No OUT_SUFFIX -> canonical resources/infer_results/{GENE_SET}_t2_t4_allpairs/.
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}

GENE_SETS=(correlation_high correlation_mid correlation_low \
           detection_high detection_mid detection_low \
           variability_high variability_mid variability_low)
GENE_SET="${GENE_SETS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] TwINFER ALL_PAIRS run on $GENE_SET (t1=2, t2=4), 20 cores, 5000 null draws, cp10k input"
# /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_infer_correlation_high.py   # [2026-09-30 replaced by env.sh variable]
ALL_PAIRS=1 INPUT_DIR=twinfer_input_cp10k GENE_SET="$GENE_SET" N_CORES=20 "$PYTHON" \
    ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/preprocessing/run_infer_correlation_high.py
echo "[$(date)] Done ($GENE_SET)"
