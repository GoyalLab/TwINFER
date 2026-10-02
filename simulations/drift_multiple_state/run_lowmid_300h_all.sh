#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --time=6:00:00
#SBATCH --job-name=lowmid_300h_all
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/lowmid_300h/logs/job-%A.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/lowmid_300h/logs/job-%A.err

# ONE job: (1) simulate the low/mid drift (recover variant) for 300 h twin time,
# 3 reps, both networks; then (2) score Step 1/2 every 10 h out to 300 h.
set -e
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python   # [2026-09-30 replaced by env.sh variable]
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export DRIFT_T1=1 DRIFT_T2=20

# SIM=/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation_variants/lowmid_300h   # [2026-09-30 replaced by env.sh variable]
SIM=${TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation_variants/lowmid_300h
# ANA=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/lowmid_300h   # [2026-09-30 replaced by env.sh variable]
ANA=${TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/lowmid_300h
mkdir -p "$SIM/logs" "$ANA/logs"
# cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/drift_multiple_state
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state

echo "=== PHASE 1: simulate (twin-time 300, recover, 3 reps x 2 networks) $(date) ==="
for NET in A_B A_to_B; do
    "$PY" regenerate_drift_variant.py \
        --network "$NET" --variant recover --twin-time 300 \
        --out-dir "$SIM" --reps 3 --jobs 3 --cores-per 4
done

echo "=== PHASE 2: score Step 1/2 every 10 h to 300 h  (N_SHUFFLES=5000) $(date) ==="
"$PY" -u plot_lowmid_100h_step12_vs_time.py \
    --sim-dir "$SIM" --out-dir "$ANA" --tmax 300 --every 10 --jobs 6

echo "=== DONE $(date) ==="
