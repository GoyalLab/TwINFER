#!/bin/bash
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16GB
#SBATCH -t 1:00:00
#SBATCH --output=/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/3f20d120-6551-47bb-b46a-4cc9df254990/scratchpad/old_code_test/slurm_%j.out
#SBATCH --error=/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/3f20d120-6551-47bb-b46a-4cc9df254990/scratchpad/old_code_test/slurm_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Diagnostic: run the ORIGINAL TwINFER-1.0 release's Gillespie engine
# (gillespie_script_variations.py, pre-propensity-floor-fix) against row 4
# (k_add=0.8) of its own bundled median_parameter.csv -- the exact config
# the release's figure_3_simulations.py used for A_rep_B. 2 reps, n_cells=6000.
# Tests whether the old engine (no floor on the combined k_on+repression
# propensity) reproduces the old data's gene_2_mRNA mean (~0.25) despite
# identical parameters to the new code's row 4 (which gives ~0.36-0.37).

# [2026-10-01 commented out: ephemeral scratchpad] 
# mkdir -p /gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/3f20d120-6551-47bb-b46a-4cc9df254990/scratchpad/old_code_test
mkdir -p "${TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/3f20d120/old_code_test"
# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"

# "$PYTHON" /gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/3f20d120-6551-47bb-b46a-4cc9df254990/scratchpad/run_old_code_test.py   # [2026-09-30 repointed from an ephemeral Claude scratchpad to the rescued copy]
"$PYTHON" ${TWINFER_PROJECT_ROOT}/clean_code/benchmarks/network_benchmarks/simulate/old_code_checks/run_old_code_test.py
