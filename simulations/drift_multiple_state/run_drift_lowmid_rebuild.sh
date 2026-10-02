#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=8:00:00
#SBATCH --job-name=drift_lowmid_rebuild
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowmid/logs/job-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowmid/logs/job-%A_%a.err

# 4 array tasks = (kind x network). Each: (1) simulate low->mid, then (2) score
# ALL TwINFER z-scores vs twin time (Step 1/2/3/4), N_SHUFFLES = 5000.
set -e
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
COMBOS=("K_frozen A_B" "K_frozen A_to_B" "K_recalc A_B" "K_recalc A_to_B")
read -r KIND NET <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

# PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python   # [2026-09-30 replaced by env.sh variable]
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export DRIFT_T1=1 DRIFT_T2=20

# SIM=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowmid/${KIND}   # [2026-09-30 replaced by env.sh variable]
SIM=${TWINFER_PROJECT_ROOT}/simulation_data/drift_lowmid/${KIND}
# ANA=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowmid/${KIND}   # [2026-09-30 replaced by env.sh variable]
ANA=${TWINFER_PROJECT_ROOT}/analysis_data/drift_lowmid/${KIND}
# mkdir -p "$SIM" "$ANA" /projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowmid/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p "$SIM" "$ANA" ${TWINFER_PROJECT_ROOT}/analysis_data/drift_lowmid/logs
# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/drift_multiple_state
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state

# network -> output type tag
case "$NET" in
  A_B)    TAG="A_B_no_reg_lowmid_${KIND}" ;;
  A_to_B) TAG="A_to_B_lowmid_${KIND}" ;;
esac

echo "=== [$SLURM_ARRAY_TASK_ID] kind=$KIND net=$NET  PHASE 1 simulate  $(date) ==="
"$PY" regenerate_drift_lowmid.py \
    --network "$NET" --kind "$KIND" --out-dir "$SIM" \
    --reps 3 --jobs 3 --cores-per 4 --twin-time 300

echo "=== [$SLURM_ARRAY_TASK_ID] PHASE 2 score all z-scores (N=5000, every 10h to 300h)  $(date) ==="
"$PY" score_lowmid_zscores_vs_time.py \
    --sim-dir "$SIM" --out-dir "$ANA" --type-tag "$TAG" \
    --tmax 300 --every 10 --ref-t1 1 --n-shuffles 5000 --jobs 3 --n-cores 4

echo "=== [$SLURM_ARRAY_TASK_ID] DONE  $(date) ==="
