#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=110GB
#SBATCH --time=4:00:00
#SBATCH --job-name=six_scen_z
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/logs/slurmLog-%A-%x.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/logs/slurmLog-%A-%x.err
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

eval "$(conda shell.bash hook)"
conda activate twinfer-code

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/drift_multiple_state
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state
# ~/.conda/envs/twinfer-code/bin/python run_6scenario_zscores.py \   # [2026-09-30 replaced by env.sh variable]
# --output-dir /projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/six_scenario \   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" run_6scenario_zscores.py \
    --output-dir ${TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/six_scenario \
    --jobs 24 --n-cores 1
