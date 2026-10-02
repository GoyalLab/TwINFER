#!/bin/bash
# Julia PIDC (NetworkInference.jl) for any expression matrix: package PIDC (symmetric) AND the directed variant.
#
#   usage:  julia_pidc_run.sh <ExpressionData.tsv> <out_dir> [threads]
#
# INPUT  <ExpressionData.tsv>: tab-separated, genes x cells; first row = cell ids (header), first column = gene names; values = expression
#        (FM06 uses log1p CP10k). Drop zero-variance genes first (the discretizer fails on them). FM06 examples are written by
#        prep_julia_pidc_fm06.py (pooled competitor input, 8,000 cells), prep_julia_D_fm06.py (the D-term input: replicate-B cells of the
#        A/B-spanning clones) and prep_julia_pidc_scaling.py (nested panels of expressed CollecTRI sources).
# OUTPUT in <out_dir>:
#   outFile.txt            package PIDC (run_pidc_julia.jl): 'Gene1 Gene2 weight', one row per ORDERED pair but the weight is symmetric
#                          (weight = cdf_i(score) + cdf_j(score), both genes' gamma-fitted score distributions)
#   outFile_directed.txt   directed PIDC (run_pidc_julia_directed.jl): 'source target weight'; the unique-information contribution is kept
#                          for [source,target] only and the weight is the gamma-CDF under the SOURCE gene's own score distribution
#                          (the same normalization as the Python PIDC in run_competitors_fatemap.py). Sanity check printed in the log:
#                          (puc + puc') equals the package's PUC matrix to ~1e-13.
#   time.txt / time_directed.txt   /usr/bin/time -v of each run (wall time, peak memory)
# Measured (8,000 cells): 65 genes 44 s / 0.44 GB, 150: 68 s / 0.54 GB, 300: 125 s / 0.72 GB, 581: 4 min 40 s / 1.45 GB (package version).
#
# ENVIRONMENT: Julia 1.10.2 (same binary as `module load julia/1.10.2`; the module command is unavailable in batch shells).
#   JULIA_DEPOT_PATH = a private writable depot first, then yscher's BEELINE depot, which already has NetworkInference 0.1.0, InformationMeasures
#   0.3.0 and LightGraphs installed (installPackages.jl in /projects/b1255/yscher/beeline_hub/Algorithms/PIDC/ lists the versions).
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
IN=${1:?input ExpressionData.tsv}; OUT=${2:?output dir}; THREADS=${3:-4}
# PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] PIPE=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline
PIPE=${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
JULIA=/software/julia/1.10.2/julia-1.10.2/usr/bin/julia
# export JULIA_DEPOT_PATH=/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/pidc_julia/julia_depot:/projects/b1255/yscher/beeline_hub/julia_depot   # [2026-09-30 replaced by env.sh variable]
export JULIA_DEPOT_PATH=${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/pidc_julia/julia_depot:/projects/b1255/yscher/beeline_hub/julia_depot
export JULIA_NUM_THREADS=$THREADS
mkdir -p "$OUT"
echo "[$(date)] package PIDC (symmetric)"
/usr/bin/time -v -o "$OUT/time.txt" $JULIA $PIPE/run_pidc_julia.jl "$IN" "$OUT/outFile.txt"
echo "[$(date)] directed PIDC"
/usr/bin/time -v -o "$OUT/time_directed.txt" $JULIA $PIPE/run_pidc_julia_directed.jl "$IN" "$OUT/outFile_directed.txt"
echo "[$(date)] done: $OUT"; grep -E "Elapsed|Maximum resident" "$OUT/time.txt" "$OUT/time_directed.txt"
