#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=fm01_ts_strat
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm01/data/twinscore_pdf_candidates_strat/logs/%x_%A_%a.log
set -eu
R=/gpfs/projects/b1255/hzhang/TwINFER_KA; PY=/home/gzu5140/.conda/envs/twinfer/bin/python
V=(all cc cx); v=${V[$SLURM_ARRAY_TASK_ID]}
export TWINFER_PROJECT_ROOT=$R OMP_NUM_THREADS=8
$PY -I $R/code/TwINFER/real_data_scripts/twinscore_pdf_candidates_strat.py --dataset FM01 \
  --candidates $R/analysis_data/fm01/data/fm01_candidates_chipatlas_epidermis_176512.tsv \
  --out $R/analysis_data/fm01/data/twinscore_pdf_candidates_strat/$v \
  --excl $R/analysis_data/fm01/data/twinscore_pdf_candidates/exclusion_list_B.json \
  --strata-csv $R/analysis_data/fm01/data/fm01_cell_covariates.csv --strata-col strat_$v
