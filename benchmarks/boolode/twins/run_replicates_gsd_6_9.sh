#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=25
#SBATCH --mem 5GB
#SBATCH -t 12:00:00
#SBATCH --array=6-9
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins/logs/gsd_rep_%A_%a.out
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/benchmarks/boolode/twins/logs/gsd_rep_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins/logs/gsd_rep_%A_%a.err
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/benchmarks/boolode/twins/logs/gsd_rep_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BOOLODE_PATH:?source clean_code/env.sh first (full BoolODE install: BoolODE/ package + data/)}"

# GSD BoolODE twin sims, replicates 6-9 (0-5 exist; replicate_6 is a stale half-written dir with only
# model.py/parameters.txt -- it gets replaced below). One array task per replicate, seed-offset = replicate
# index (same convention as run_replicates.sh). GSD is a built-in network in twin_similarity_sweep.py
# (GSD.txt + GSD_ics.txt, simulation_time=8, single branch fraction 0.5). The earlier GSD replicates took
# ~3.3 h each at 25 CPUs / 5 GB, so 12 h per task is ample.
# SCRIPT_DIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_CODE_ROOT}/benchmarks/boolode/twins
# PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python3}"
# OUTPUT_BASE=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/boolode_sims_replicates   # [2026-09-30 replaced by env.sh variable]
OUTPUT_BASE=${TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims_replicates
NETWORK=GSD
REP=${SLURM_ARRAY_TASK_ID}

STAGING_DIR="${OUTPUT_BASE}/_staging_gsd_${SLURM_ARRAY_JOB_ID:-manual}_${REP}"
FINAL_DIR="${OUTPUT_BASE}/${NETWORK}/replicate_${REP}"
echo "=== ${NETWORK} replicate ${REP} (seed-offset=${REP}) -> ${FINAL_DIR}/ ==="
"$PYTHON" "${SCRIPT_DIR}/twin_similarity_sweep.py" \
    --network "$NETWORK" --n-pairs 6000 --workers 25 --seed-offset "$REP" \
    --output-dir "$STAGING_DIR"
rm -rf "$FINAL_DIR"
mv "${STAGING_DIR}/${NETWORK}" "$FINAL_DIR"
rmdir "$STAGING_DIR"
echo "Done: ${NETWORK} replicate ${REP}"
