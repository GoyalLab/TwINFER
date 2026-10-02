#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=90GB
#SBATCH --time=2:00:00
#SBATCH --job-name=regen_drift_AtoB
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation/regen_AtoB_slurm-%A.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation/regen_AtoB_slurm-%A.err
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

eval "$(conda shell.bash hook)"
conda activate twinfer-code

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/drift_multiple_state
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state
# 10 processes x 4 cores each = 40
# ~/.conda/envs/twinfer-code/bin/python regenerate_drift_A_to_B.py \   # [2026-09-30 replaced by env.sh variable]
# --out-dir /projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation \   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" regenerate_drift_A_to_B.py \
    --out-dir ${TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation \
    --reps 20 --jobs 10 --cores-per 4
