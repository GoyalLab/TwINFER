#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem 32GB
#SBATCH -t 2:00:00
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/cleanup_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/cleanup_%j.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# Cleanup network_sweep_final's simulation_data per the finalized 150-file
# manifest: gzip+verify the 150 before_division checkpoints that correspond
# to kept runs (~448GB uncompressed), delete the 106 before_division files
# that don't (retry waste, ~329GB), and delete the 100 excess complete
# simulation files that weren't chosen (~5.5GB). Nothing is deleted until its
# .gz replacement is verified (gzip -t + exact decompressed-size check), or
# until confirmed to have no bearing on the kept manifest.

# /home/gzu5140/.conda/envs/twinfer-code/bin/python3 /home/gzu5140/TwINFER_KA/code/Beeline/cleanup_network_sweep_final.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}" ${TWINFER_CODE_ROOT}/benchmarks/beeline/scripts/cleanup_network_sweep_final.py
