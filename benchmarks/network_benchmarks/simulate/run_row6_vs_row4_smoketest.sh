#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16GB
#SBATCH -t 1:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/_row6_vs_row4_smoketest/slurm_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/_row6_vs_row4_smoketest/slurm_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Diagnostic only (see test_row6_vs_row4_smoketest.py docstring): 2 reps each
# at median_parameter.csv row 4 (k_add=0.8) and row 6 (k_add=2), current
# simulation code, to determine which row's gene_2 mRNA mean matches the old
# figure_3_simulations/A_rep_B data (~0.25) vs the new figure_3_1k data
# (~0.36). n_cores=8 -> expect roughly 4-8 min/rep based on the documented
# n_cores=4 (512s/rep) vs n_cores=20 (133s/rep) benchmarks; 4 reps total,
# 1h budget leaves large margin.

# mkdir -p /home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/_row6_vs_row4_smoketest   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/causal_direction_inference/_row6_vs_row4_smoketest
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/simulate"

"$PYTHON" test_row6_vs_row4_smoketest.py
