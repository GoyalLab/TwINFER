#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem 48GB
#SBATCH -t 12:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/boolode_sims_real_networks/logs/allpairs_%x_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/boolode_sims_real_networks/logs/allpairs_%x_%j.err
set -uo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# ALL_PAIRS TwINFER inference on the BoolODE twin sims for the networks given as arguments, all their
# replicates, 4 replicates at a time x 8 cores each (same layout as the Gillespie run_real_network_allpairs.sh).
# Submitted as 3 separate jobs (no array):
#   sbatch --job-name=bo_pluri run_boolode_real_infer_allpairs.sh Pluripotent_real
#   sbatch --job-name=bo_emt_gsd run_boolode_real_infer_allpairs.sh EMT_real GSD
#   sbatch --job-name=bo_small run_boolode_real_infer_allpairs.sh B_cell_activation HSC mCAD VSC
# GSD replicates 6-9 don't exist until job 7196023 finishes -> they report FAILED here and are rerun later
# (existing outputs are skipped). Settings: see boolode_real_infer_allpairs.py.
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/infer"
for NET in "$@"; do
    echo "[$(date)] ${NET}"
    "$PYTHON" boolode_real_infer_allpairs.py --only "$NET" --n-jobs 4 --n-cores-per-task 8 2>&1 \
        | grep -E "tasks|rep .*: (OK|SKIPPED|FAILED)"
done
echo "[$(date)] done: $*"
