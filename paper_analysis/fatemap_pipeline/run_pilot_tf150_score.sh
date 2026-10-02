#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 64GB
#SBATCH -t 24:00:00
#SBATCH --job-name=pilot150_score
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/pilot150_score_%j.out
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/pilot150_score_%j.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/pilot150_score_%j.err
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/pilot150_score_%j.err
# PILOT scoring: (1) directed Julia PIDC on the replicate-B cells (D term), (2) final gated+bootstrap TwinScore_supplement with the PARALLEL bootstraps
# (n_cores=8) and the directed Julia D. Needs z_dagger_fm06_tf_pilot150_absplit_allpairs.json from run_pilot_tf150_infer.sh.
# Output: /projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data/twinscore_supp_gated_bootstrap_pilot/
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
GS=tf_pilot150
# PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] PIPE=${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline
PIPE=${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
# DATA=/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/data   # [2026-09-30 replaced by env.sh variable]
DATA=${TWINFER_PROJECT_ROOT}/analysis_data/fm06/data
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
cd $PIPE
echo "[$(date)] (1) D input + Julia PIDC"
"$PYTHON" prep_julia_D_fm06.py $GS
./julia_pidc_run.sh $DATA/pidc_julia/D_$GS/ExpressionData.csv $DATA/pidc_julia/D_$GS 8
echo "[$(date)] (2) scoring (parallel bootstraps, directed Julia D)"
TWINFER_GB=apply_twinscore_supplement_fatemap_gated_bootstrap_parallel "$PYTHON" apply_twinscore_supplement_fatemap_gated_bootstrap_juliaD.py FM06 --gene-set $GS \
    --n-shuffles 2000 --n-cores 8 --absplit --d-file $DATA/pidc_julia/D_$GS/outFile_directed.txt --out-dir $DATA/twinscore_supp_gated_bootstrap_pilot
echo "[$(date)] Done"
