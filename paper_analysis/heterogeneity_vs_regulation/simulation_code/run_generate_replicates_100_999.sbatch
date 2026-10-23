#!/bin/bash
# SLURM array job: replicates 100-999 of the six single-state / two-state configurations (heterogeneity_vs_regulation figure).
#
# What it does : each of the 5,400 array tasks (6 configurations x 900 replicates) runs ONE replicate with simulate_heterogeneity_and_regulation.py
#                  CONFIG_INDEX = task / 900   (0 A_B, 1 A_B_low_k_on, 2 A_B_high_k_on, 3 A_to_B, 4 A_to_B_low_k_on, 5 A_to_B_high_k_on)
#                  REP_START    = 100 + task % 900   (replicate number 100-999)
#                The simulation settings are those of simulate_heterogeneity_and_regulation.py. Together with
#                run_generate_replicates_0_99.sbatch (replicates 0-99) this gives 1,000 replicates per configuration.
# Output       : $OUT_DIR/<config>/ (default final_figure_data/heterogeneity_vs_regulation/simulation_data/single_and_two_state_simulations);
#                the six <config> sub-folders must exist and be writable before submitting. Logs: simulation_data/logs/gensim1000_<array>_<task>.log
#                About 200 MB per replicate (about 1.1 TB for the whole array).
# Resources    : 20 CPUs (matches NCORES / numba threads below), 3 GB, 1 h per task; at most 150 tasks run at once (--array=0-5399%150).
#                (The original run requested 12 CPUs with 20 numba threads; corrected here.)
# Usage        : source code/TwINFER/env.sh          # sets TWINFER_CODE_ROOT, TWINFER_PYTHON, ...
#                sbatch run_generate_replicates_100_999.sbatch
#
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-5399%150
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --job-name=gensim1000
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/logs/gensim1000_%A_%a.log

set -euo pipefail
# PROJ="/gpfs/projects/b1255/yscher/Transcriptomic Distance"   # original (yscher's tree)
# cd "$PROJ"
# mkdir -p exports/handoff_logs
# Source code/TwINFER/env.sh before sbatch (sets TWINFER_PROJECT_ROOT, TWINFER_CODE_ROOT, TWINFER_PYTHON)
: "${TWINFER_CODE_ROOT:?source code/TwINFER/env.sh before submitting}"
cd "${TWINFER_CODE_ROOT}/paper_analysis/heterogeneity_vs_regulation/simulation_code"

# 6 configurations x 900 new replicates (100..999) = 5,400 tasks.
export SET_SIZE=1000
export CONFIG_INDEX=$(( SLURM_ARRAY_TASK_ID / 900 ))
export REP_START=$(( 100 + SLURM_ARRAY_TASK_ID % 900 ))
export REP_COUNT=1

export NCORES=20
export NUMBA_NUM_THREADS=20
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export PYTHONUNBUFFERED=1
# export TWINFER_REPO=TwINFER_tip   # original: pristine upstream worktree at b4bb9cb; now the pip-installed twinfer package is used
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_gen1000_${SLURM_ARRAY_TASK_ID}"
export MPLBACKEND=Agg

# PY=/hpc/software/mamba/24.3.0/bin/python3   # original interpreter + setup_env.sh (pip --user installs); replaced by the twinfer-code env from env.sh
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
[ -x "$PY" ] || { echo "interpreter missing: $PY"; exit 1; }

echo "task ${SLURM_ARRAY_TASK_ID}: configuration ${CONFIG_INDEX}, replicate ${REP_START}"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  started $(date)"
"$PY" simulate_heterogeneity_and_regulation.py
echo "finished $(date)"
