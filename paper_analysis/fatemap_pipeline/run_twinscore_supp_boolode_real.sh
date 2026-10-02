#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem 48GB
#SBATCH -t 12:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/boolode_sims_real_networks/logs/twinscore_%x_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/boolode_sims_real_networks/logs/twinscore_%x_%j.err
set -uo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# TwinScore_supplement (gated + bootstrapped) on the BoolODE twin sims, for the networks given as arguments, all
# replicates that have BOTH a converted sim file and an all-pairs inference JSON (z_dagger source) -- others are
# skipped, so this can be rerun later for GSD 6-9. 4 replicates at a time x 8 cores. Existing outputs are skipped.
# Submitted with --dependency=afterok on the matching inference job.
# ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
ROOT=${TWINFER_PROJECT_ROOT}
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd $ROOT/code/TwINFER/paper_analysis/fatemap_pipeline
cd ${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline
OUT=$ROOT/analysis_data/boolode_sims_real_networks/twinscore_supp_gated_bootstrap
mkdir -p "$OUT"
TASKS=$(mktemp)
for NET in "$@"; do
    for f in $ROOT/analysis_data/boolode_sims_real_networks/twinfer_format/$NET/replicate_*_simulation.csv; do
        REP=$(basename "$f" | sed 's/replicate_\([0-9]*\)_simulation.csv/\1/')
        [ -f "$ROOT/analysis_data/boolode_sims_real_networks/twinfer_inference_allpairs/${NET}_rep_${REP}_all_results.json" ] || { echo "skip ${NET} rep ${REP}: no inference json"; continue; }
        [ -f "$OUT/${NET}_rep_${REP}_metrics.json" ] && { echo "skip ${NET} rep ${REP}: done"; continue; }
        echo "$NET $REP" >> "$TASKS"
    done
done
echo "[$(date)] $(wc -l < "$TASKS") tasks"
cat "$TASKS" | xargs -P 4 -L 1 bash -c '"'"$PYTHON"'" apply_twinscore_supplement_sim_gated_bootstrap.py $0 $1 --n-cores 8 2>&1 | grep -E "^\[.* rep .*\]|Error|Traceback" | cut -c1-600'
echo "[$(date)] done: $*"
