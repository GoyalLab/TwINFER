#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=90GB
#SBATCH --time=3:00:00
#SBATCH --job-name=lowmid_100h_z
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/lowmid_100h/logs/analysis-%A.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/lowmid_100h/logs/analysis-%A.err
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
eval "$(conda shell.bash hook)"; conda activate twinfer-code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4 DRIFT_T1=1 DRIFT_T2=20
# mkdir -p /projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/lowmid_100h/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/lowmid_100h/logs
# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/drift_multiple_state
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state
# ~/.conda/envs/twinfer-code/bin/python -u plot_lowmid_100h_step2_vs_time.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" -u plot_lowmid_100h_step2_vs_time.py
