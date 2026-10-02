#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem 16GB
#SBATCH -t 01:00:00
#SBATCH --job-name=pulse_chase_mcad
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/pulse_and_chase/slurm_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/pulse_and_chase/slurm_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# mCAD pulse-and-chase local linear response (see pulse_chase.py docstring).
# Env python directly -- conda shell hook is unreliable in batch.
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# HERE=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/pulse_and_chase   # [2026-09-30 replaced by env.sh variable]
HERE=${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/pulse_and_chase
cd "$HERE"

export N_CELLS=12000          # CRN-paired; per-cell noise cancels in pert-ctrl
export T_BURN=500             # protein t1/2 = 45 h  -> ~11 half-lives, fully equilibrated
export TAUS=1,3,6,12,24,48
export EPS=0.10               # +10% k_prod_mRNA, K held at WT calibration
export N_CORES=20
export NUMBA_NUM_THREADS=20
export OMP_NUM_THREADS=20

echo "[$(date)] pulse_chase.py  N_CELLS=$N_CELLS T_BURN=$T_BURN TAUS=$TAUS EPS=$EPS"
"$PYTHON" -u pulse_chase.py
echo "[$(date)] done"
