#!/bin/bash
# SLURM array job: replicates 0-99 of the six single-state / two-state configurations (heterogeneity_vs_regulation figure).
#
# What it does : each of the 600 array tasks (6 configurations x 100 replicates) runs ONE replicate with simulate_heterogeneity_and_regulation.py
#                  CONFIG_INDEX = task / 100   (0 A_B, 1 A_B_low_k_on, 2 A_B_high_k_on, 3 A_to_B, 4 A_to_B_low_k_on, 5 A_to_B_high_k_on)
#                  REP_START    = task % 100   (replicate number 0-99)
#                The simulation settings are those of simulate_heterogeneity_and_regulation.py.
# Output       : $OUT_DIR/<config>/ (default final_figure_data/heterogeneity_vs_regulation/simulation_data/single_and_two_state_simulations);
#                the six <config> sub-folders must exist and be writable before submitting. Logs: simulation_data/logs/gensim100_<array>_<task>.log
# Resources    : 20 CPUs, 8 GB, 8 h per task (a replicate took 41-456 s, median 190 s, on 20 cores).
# Usage        : source code/TwINFER/env.sh          # sets TWINFER_CODE_ROOT, TWINFER_PYTHON, ...
#                sbatch run_generate_replicates_0_99.sbatch
#                (replicates 100-999 are run by run_generate_replicates_100_999.sbatch)
#
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-599
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=gensim100
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/logs/gensim100_%A_%a.log

set -euo pipefail
# PROJ="/gpfs/projects/b1255/yscher/Transcriptomic Distance"   # original (yscher's tree)
# cd "$PROJ"
# mkdir -p exports/handoff_logs
# Source code/TwINFER/env.sh before sbatch (sets TWINFER_PROJECT_ROOT, TWINFER_CODE_ROOT, TWINFER_PYTHON)
: "${TWINFER_CODE_ROOT:?source code/TwINFER/env.sh before submitting}"
cd "${TWINFER_CODE_ROOT}/paper_analysis/heterogeneity_vs_regulation/simulation_code"

# 600 tasks = 6 configurations x 100 replicates. Task id -> (configuration, replicate).
export SET_SIZE=100
export CONFIG_INDEX=$(( SLURM_ARRAY_TASK_ID / 100 ))
export REP_START=$(( SLURM_ARRAY_TASK_ID % 100 ))
export REP_COUNT=1

export NCORES=20
export NUMBA_NUM_THREADS=20
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export PYTHONUNBUFFERED=1
# export TWINFER_REPO=TwINFER_tip   # original: pristine upstream worktree at b4bb9cb; now the pip-installed twinfer package is used
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_gen100_${SLURM_ARRAY_TASK_ID}"
export MPLBACKEND=Agg

# PY=/hpc/software/mamba/24.3.0/bin/python3   # original interpreter + setup_env.sh (pip --user installs); replaced by the twinfer-code env from env.sh
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
[ -x "$PY" ] || { echo "interpreter missing: $PY"; exit 1; }

echo "task ${SLURM_ARRAY_TASK_ID}: configuration ${CONFIG_INDEX}, replicate ${REP_START}"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  started $(date)"
"$PY" simulate_heterogeneity_and_regulation.py
echo "finished $(date)"
