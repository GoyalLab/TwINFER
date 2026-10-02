#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 96GB
#SBATCH -t 12:00:00
#SBATCH --job-name=pidc_julia_scale
#SBATCH --array=0-3
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/pidc_julia_scale_%A_%a.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/pidc_julia_scale_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/pidc_julia_scale_%A_%a.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/pidc_julia_scale_%A_%a.err
# Julia PIDC scaling test: nested panels of N = 65, 150, 300, 581 CollecTRI-source genes (>5% detection), 8,000 cells, one task per N.
# Wall time and peak memory of each run go to <panel>/time.txt (/usr/bin/time -v). Each task first builds its own expression table with prep_julia_pidc_scaling.py.
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
SIZES=(65 150 300 581)
N=${SIZES[$SLURM_ARRAY_TASK_ID]}
# PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] PIPE=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline
PIPE=${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
# OUT=/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/pidc_julia/scaling_N$N   # [2026-09-30 replaced by env.sh variable]
OUT=${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/pidc_julia/scaling_N$N
JULIA=/software/julia/1.10.2/julia-1.10.2/usr/bin/julia
# export JULIA_DEPOT_PATH=/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/pidc_julia/julia_depot:/projects/b1255/yscher/beeline_hub/julia_depot   # [2026-09-30 replaced by env.sh variable]
export JULIA_DEPOT_PATH=${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/pidc_julia/julia_depot:/projects/b1255/yscher/beeline_hub/julia_depot
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
# /home/gzu5140/.conda/envs/twinfer/bin/python3 $PIPE/prep_julia_pidc_scaling.py $N   # writes $OUT/ExpressionData.csv (loads the QC counts, ~2 GB)   # [2026-09-30 commented out: env `twinfer` replaced by `twinfer-code` (user decision)]
"${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}" $PIPE/prep_julia_pidc_scaling.py $N   # writes $OUT/ExpressionData.csv (loads the QC counts, ~2 GB)
echo "[$(date)] N=$N julia PIDC start"
/usr/bin/time -v -o "$OUT/time.txt" $JULIA $PIPE/run_pidc_julia.jl "$OUT/ExpressionData.csv" "$OUT/outFile.txt"
echo "[$(date)] N=$N done"; grep -E "Elapsed|Maximum resident" "$OUT/time.txt"
