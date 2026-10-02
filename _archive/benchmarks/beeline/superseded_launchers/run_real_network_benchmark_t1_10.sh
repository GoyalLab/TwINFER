#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 32GB
#SBATCH -t 12:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference_t1_10/logs/benchmark_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference_t1_10/logs/benchmark_%j.err
set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"

# t1=10,t2=20 rerun of run_real_network_benchmark.sh (65 replicates / 7 networks):
# PIDC/PPCOR/SCODE/SCSGL/PEARSON + GENIE3/GRNBOOST2 bundled in one config, over the
# twin_paired-only real_networks_t1_10 input root. Wall time trimmed from 24h to 12h since this
# root has ONLY twin_paired (no spread duplicate). Regenerate the config first if the dataset set
# changed: python generate_t1_10_beeline_configs.py
#
# Usage: sbatch run_real_network_benchmark_t1_10.sh

# mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/beeline_inference_t1_10/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference_t1_10/logs
module load julia/1.10.2 R/4.2.3 ruby/3.1.0-gcc-4.8.5
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/Beeline   # [2026-09-30 replaced by env.sh variable]
cd ${TWINFER_BEELINE_PATH}

# BEELINE_PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python   # [2026-09-30 replaced by env.sh variable]
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"

echo "[$(date)] Starting real_networks t1=10 benchmark (B_cell_activation/Circadian_cycle/GSD/HSC/mCAD/Pluripotent/VSC)..."
echo "Using python: $BEELINE_PYTHON"
"$BEELINE_PYTHON" BLRunner.py --config config-files/config_real_networks_t1_10.yaml --reverse --yes --skip-populated
status=$?
echo "[$(date)] real_networks t1=10 benchmark finished with exit code $status"
