#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=8:00:00
#SBATCH --job-name=drift_lowmid_v2
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowmid/logs/v2-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowmid/logs/v2-%A_%a.err

# 5 array tasks: 2x K_ramp (live mean-field K, reuse K_frozen burn-ins) + 3x single-state.
# Each: simulate 3 reps -> score ALL z-scores vs twin time (N_SHUFFLES=5000, every 10h to 300h).
set -e
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python   # [2026-09-30 replaced by env.sh variable]
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export DRIFT_T1=1 DRIFT_T2=20

# BASE=/projects/b1255/hzhang/TwINFER_KA   # [2026-09-30 replaced by env.sh variable]
BASE=${TWINFER_PROJECT_ROOT}
KFROZEN=$BASE/simulation_data/drift_lowmid/K_frozen
mkdir -p $BASE/analysis_data/drift_lowmid/logs
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] cd $BASE/code/TwINFER/drift_multiple_state
cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state

case $SLURM_ARRAY_TASK_ID in
  0) MODE=kramp;  NET=A_B;    TAG=A_B_no_reg_lowmid_K_ramp ;;
  1) MODE=kramp;  NET=A_to_B; TAG=A_to_B_lowmid_K_ramp ;;
  2) MODE=single; NET=A_B;    KCAL=0.66; TAG=A_B_no_reg_single_Kcalib_0p66 ;;
  3) MODE=single; NET=A_to_B; KCAL=0.66; TAG=A_to_B_single_Kcalib_0p66 ;;
  4) MODE=single; NET=A_to_B; KCAL=0.12; TAG=A_to_B_single_Kcalib_0p12 ;;
esac

if [ "$MODE" = kramp ]; then
    SIM=$BASE/simulation_data/drift_lowmid/K_ramp
    ANA=$BASE/analysis_data/drift_lowmid/K_ramp
    mkdir -p "$SIM" "$ANA"
    echo "=== [$SLURM_ARRAY_TASK_ID] K_ramp $NET  simulate  $(date) ==="
    "$PY" regenerate_drift_lowmid.py --network "$NET" --kind K_ramp --out-dir "$SIM" \
        --kfrozen-dir "$KFROZEN" --reps 3 --jobs 3 --cores-per 4 \
        --twin-time 300 --k-meanfield-dt 0.05
else
    SIM=$BASE/simulation_data/drift_lowmid/single_2K
    ANA=$BASE/analysis_data/drift_lowmid/single_2K
    mkdir -p "$SIM" "$ANA"
    echo "=== [$SLURM_ARRAY_TASK_ID] single $NET K@$KCAL  simulate  $(date) ==="
    "$PY" regenerate_single_2K.py --network "$NET" --k-calib "$KCAL" --out-dir "$SIM" \
        --reps 3 --jobs 3 --cores-per 4 --twin-time 300
fi

echo "=== [$SLURM_ARRAY_TASK_ID] score $TAG  (N=5000, every 10h to 300h)  $(date) ==="
"$PY" score_lowmid_zscores_vs_time.py --sim-dir "$SIM" --out-dir "$ANA" --type-tag "$TAG" \
    --tmax 300 --every 10 --ref-t1 1 --n-shuffles 5000 --jobs 3 --n-cores 4
echo "=== [$SLURM_ARRAY_TASK_ID] DONE  $(date) ==="
