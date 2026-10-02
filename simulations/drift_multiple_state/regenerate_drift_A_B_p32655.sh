#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30GB
#SBATCH --time=4:00:00
#SBATCH --job-name=regen_drift_AB_p3
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation/regen_AB_p32655-%A.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation/regen_AB_p32655-%A.err
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
# Shared output dir with the b1042 job (5374312) and the login-node run;
# --skip-existing (default) means the three cooperate -- each grabs reps not
# yet written. Interleaving is fine.
# ~/.conda/envs/twinfer-code/bin/python regenerate_drift_A_B.py \   # [2026-09-30 replaced by env.sh variable]
# --out-dir /projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation \   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" regenerate_drift_A_B.py \
    --out-dir ${TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation \
    --reps 20 --jobs 5 --cores-per 4
