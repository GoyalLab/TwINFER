#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --mem=50G
#SBATCH -t 48:00:00
#SBATCH --job-name=pluripotent_6000steps
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/Pluripotent/simulate/6000steps_slurm_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/Pluripotent/simulate/6000steps_slurm_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Vanilla (non-multistate) Pluripotent rerun at 6000 before-division steps
# (was 1000 -- settling-time analysis showed the 1000-step run had not
# reached steady state). 1 replicate only; conda hook is unreliable in
# batch, so call the env python directly (twinfer-code env, per project
# convention).
# PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python   # [2026-09-30 replaced by env.sh variable]
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
export PYTHONUNBUFFERED=1

# SCRIPT_DIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR="${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/simulate"
cd "$SCRIPT_DIR"

echo "[$(date)] job $SLURM_JOB_ID on $(hostname)"
"$PY" -V
"$PY" -u pluripotent_sim_6000steps.py --config_index 0
echo "[$(date)] done (exit $?)"
