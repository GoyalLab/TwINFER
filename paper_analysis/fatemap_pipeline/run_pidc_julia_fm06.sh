#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16GB
#SBATCH -t 2:00:00
#SBATCH --job-name=pidc_julia_fm06
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/pidc_julia_fm06_%j.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/pidc_julia_fm06_%j.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/pidc_julia_fm06_%j.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/pidc_julia_fm06_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
GS=${1:-correlation_high}
# PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] PIPE=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline
PIPE=${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
# OUT=/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/pidc_julia/$GS   # [2026-09-30 replaced by env.sh variable]
OUT=${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/pidc_julia/$GS
# PYTHON=/home/gzu5140/.conda/envs/twinfer/bin/python3   # [2026-09-30 commented out: env `twinfer` replaced by `twinfer-code` (user decision)]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
JULIA=/software/julia/1.10.2/julia-1.10.2/usr/bin/julia   # same binary as `module load julia/1.10.2` (module command is unavailable in batch shells)
# private writable depot first, yscher's BEELINE depot (has NetworkInference 0.1.0, LightGraphs, InformationMeasures) second
# export JULIA_DEPOT_PATH=/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/pidc_julia/julia_depot:/projects/b1255/yscher/beeline_hub/julia_depot   # [2026-09-30 replaced by env.sh variable]
export JULIA_DEPOT_PATH=${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/pidc_julia/julia_depot:/projects/b1255/yscher/beeline_hub/julia_depot
# mkdir -p /projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/pidc_julia/julia_depot   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/pidc_julia/julia_depot
"$PYTHON" $PIPE/prep_julia_pidc_fm06.py "$GS"
echo "[$(date)] julia PIDC start"
/usr/bin/time -v -o "$OUT/time.txt" $JULIA $PIPE/run_pidc_julia.jl "$OUT/ExpressionData.csv" "$OUT/outFile.txt"
echo "[$(date)] julia PIDC done"; grep -E "Elapsed|Maximum resident" "$OUT/time.txt"
