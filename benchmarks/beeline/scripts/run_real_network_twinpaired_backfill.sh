#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 32GB
#SBATCH -t 24:00:00
#SBATCH --job-name=real_network_tp_backfill
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/benchmarking_analysis/twinfer_real/outputs/logs/backfill_%j.out   # [2026-10-01 was /scratch/gzu5140/twinfer_real/...; #SBATCH cannot use variables]
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/benchmarking_analysis/twinfer_real/outputs/logs/backfill_%j.err
set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Backfills the missing/incomplete twin_paired BEELINE runs for the curated
# real-world networks (GSD: 0/63, HSC: 0/70, mCAD: 35/70 rankedEdges.csv present;
# VSC twin_paired is already complete at 70/70). --skip-populated (checks
# rankedEdges.csv existence, not just working_dir non-emptiness) means every
# already-complete run/algorithm -- all of spread, and VSC's twin_paired -- is
# skipped automatically, so this is safe to run against the whole dataset set.

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] mkdir -p /scratch/gzu5140/twinfer_real/outputs/logs
mkdir -p "${TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_real/outputs/logs"

module load julia/1.10.2 R/4.2.3 ruby/3.1.0-gcc-4.8.5

# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_BEELINE_PATH}

# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

echo "[$(date)] Starting twin_paired backfill (GSD/HSC/mCAD/VSC)..."
echo "Using python: $BEELINE_PYTHON"
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_real_network_twinpaired_backfill.yaml --reverse --yes --skip-populated
status=$?
echo "[$(date)] Backfill finished with exit code $status"
