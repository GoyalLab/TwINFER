#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=90GB
#SBATCH --time=3:00:00
#SBATCH --job-name=drift_variant
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation_variants/logs/slurm-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation_variants/logs/slurm-%A_%a.err
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# 4 array tasks: (network x variant)
#   0: A_B     fast5h    3: A_to_B  recover   ... see COMBOS below
COMBOS=("A_B fast5h" "A_to_B fast5h" "A_B recover" "A_to_B recover")
read -r NETWORK VARIANT <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

eval "$(conda shell.bash hook)"
conda activate twinfer-code

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# OUT=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation_variants   # [2026-09-30 replaced by env.sh variable]
OUT=${TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation_variants
mkdir -p "$OUT/logs"

# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/drift_multiple_state
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state
echo "task $SLURM_ARRAY_TASK_ID  network=$NETWORK  variant=$VARIANT"
# 10 processes x 4 cores each = 40
# ~/.conda/envs/twinfer-code/bin/python regenerate_drift_variant.py \   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" regenerate_drift_variant.py \
    --network "$NETWORK" --variant "$VARIANT" \
    --out-dir "$OUT" \
    --reps 20 --jobs 10 --cores-per 4
