#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48GB
#SBATCH --time=3:00:00
#SBATCH --job-name=drift_lowmid_100h
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation_variants/lowmid_100h/logs/slurm-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation_variants/lowmid_100h/logs/slurm-%A_%a.err
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# "low/mid drift" = the recover variant (up: k_on 0.12x -> 1.0x baseline; down: stays 0.12x)
# run 100 h of twins, hourly, 3 replicates, both networks
NETS=("A_B" "A_to_B")
NETWORK="${NETS[$SLURM_ARRAY_TASK_ID]}"

eval "$(conda shell.bash hook)"
conda activate twinfer-code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# OUT=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation_variants/lowmid_100h   # [2026-09-30 replaced by env.sh variable]
OUT=${TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation_variants/lowmid_100h
mkdir -p "$OUT/logs"

# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/drift_multiple_state
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state
echo "network=$NETWORK  variant=recover  twin-time=100  reps=3"
# ~/.conda/envs/twinfer-code/bin/python regenerate_drift_variant.py \   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" regenerate_drift_variant.py \
    --network "$NETWORK" --variant recover --twin-time 100 \
    --out-dir "$OUT" --reps 3 --jobs 3 --cores-per 4
