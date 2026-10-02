#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 8:00:00
#SBATCH --job-name=fm06_juliaD_dir
#SBATCH --array=0-8
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/fm06_juliaD_dir_%A_%a.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/fm06_juliaD_dir_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/fm06_juliaD_dir_%A_%a.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/fm06_juliaD_dir_%A_%a.err
# Rescoring TwinScore_supplement with the DIRECTED Julia PIDC as the D term. Needs pidc_julia/D_<gs>/outFile_directed.txt (made by
# julia_pidc_run.sh / run_pidc_julia_directed.jl on the replicate-B cells; prep_julia_D_fm06.py writes the input). Everything else is the final script.
# Output: /projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/twinscore_supp_gated_bootstrap_juliaD_directed/
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
GENE_SETS=(variability_high variability_mid variability_low detection_high detection_mid detection_low correlation_high correlation_mid correlation_low)
GS=${GENE_SETS[$SLURM_ARRAY_TASK_ID]}
# PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] PIPE=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline
PIPE=${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
# DATA=/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data   # [2026-09-30 replaced by env.sh variable]
DATA=${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
cd $PIPE
echo "[$(date)] $GS: rescoring with the directed Julia D"
"$PYTHON" apply_twinscore_supplement_fatemap_gated_bootstrap_juliaD.py FM06 --gene-set "$GS" --n-shuffles 2000 --n-cores 8 --absplit \
    --d-file $DATA/pidc_julia/D_$GS/outFile_directed.txt --out-dir $DATA/twinscore_supp_gated_bootstrap_juliaD_directed
echo "[$(date)] Done"
