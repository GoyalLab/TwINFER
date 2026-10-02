#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -J real_net_full
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/network_sweep_final/slurm_real_networks_%j.out
#SBATCH -e /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/network_sweep_final/slurm_real_networks_%j.err
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/network_sweep_final   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final
# /home/gzu5140/.conda/envs/twinfer-code/bin/python3 run_full_formula_real_networks.py   # [2026-09-30 replaced by env.sh variable]
# "${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}" run_full_formula_real_networks.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}" run_full_formula_real_networks.py
