#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --job-name=8scen_zscores
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/eight_scenario/logs/%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/eight_scenario/logs/%A_%a.err

# 8 scenarios x {20 reps baseline/multistate, 3 reps drift_lowmid} at two (t1,t2)
# settings: task 0 -> t1=10,t2=20 ; task 1 -> t1=1,t2=20.
set -e
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python   # [2026-09-30 replaced by env.sh variable]
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
# BASE=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/eight_scenario   # [2026-09-30 replaced by env.sh variable]
BASE=${TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario
mkdir -p "$BASE/logs"
# cd /home/gzu5140/TwINFER_KA/code/TwINFER/drift_multiple_state   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/simulations/drift_multiple_state"

case $SLURM_ARRAY_TASK_ID in
  0) export DRIFT_T1=10 DRIFT_T2=20; OUT=$BASE/t1_10_t2_20 ;;
  1) export DRIFT_T1=1  DRIFT_T2=20; OUT=$BASE/t1_1_t2_20 ;;
esac
mkdir -p "$OUT"

echo "=== [$SLURM_ARRAY_TASK_ID] t1=$DRIFT_T1 t2=$DRIFT_T2 -> $OUT  $(date) ==="
"$PY" run_6scenario_zscores.py --output-dir "$OUT" --n-cores 4 --jobs 6
echo "=== [$SLURM_ARRAY_TASK_ID] DONE  $(date) ==="
