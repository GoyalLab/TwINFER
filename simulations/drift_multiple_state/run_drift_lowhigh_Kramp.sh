#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --job-name=drift_lowhigh
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowhigh_Kramp/logs/%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowhigh_Kramp/logs/%A_%a.err

# Drift 0.66 -> {0.12, 1.66} over 0-10 h with live mean-field K, 1000 replicates per network.
# Burn-in end states reused from the Figure-2 1000-replicate set (same parents as single-state rep n).
# 3 array tasks: 0 = A_B reps 0-999 ; 1 = A_to_B reps 0-499 ; 2 = A_to_B reps 500-999.
# Each: 12 parallel reps x 4 numba threads. Re-submitting skips replicates whose output already exists.
# Source clean_code/env.sh before sbatch (sbatch exports the environment).
set -e
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before submitting}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before submitting}"
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state
case $SLURM_ARRAY_TASK_ID in
  0) NET=A_B;    REPS=0-999 ;;
  1) NET=A_to_B; REPS=0-499 ;;
  2) NET=A_to_B; REPS=500-999 ;;
esac
OUT=${TWINFER_PROJECT_ROOT}/simulation_data/drift_lowhigh_Kramp/$NET
echo "=== [$SLURM_ARRAY_TASK_ID] $NET reps $REPS  $(date) ==="
"$PY" regenerate_drift_lowhigh.py --network "$NET" --out-dir "$OUT" --reps "$REPS" \
    --jobs 12 --cores-per 4 --twin-time 48 --tau 10 --burnin-hours 1000 --dt-K 0.05
echo "=== [$SLURM_ARRAY_TASK_ID] DONE  $(date) ==="
