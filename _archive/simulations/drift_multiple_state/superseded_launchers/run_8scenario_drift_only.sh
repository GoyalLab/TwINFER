#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=1:00:00
#SBATCH --job-name=8scen_drift_only
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/eight_scenario/logs/drift_only-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_inference/eight_scenario/logs/drift_only-%A_%a.err

# Login-node run OOM'd (12 concurrent loky workers each loading a ~125MB/6000-cell
# drift_lowmid CSV blew through the shared node's ~21GB free). Redo on SLURM with
# dedicated memory. Only the 4 new drift_lowmid scenarios (12 rep-tasks each) --
# no_regulation/A_to_B/multistate_A_B/multistate_A_to_B already scored (archived).
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

DRIFT_ONLY="kramp_A_B:1,kramp_A_B:2,kramp_A_B:3,kramp_A_to_B:1,kramp_A_to_B:2,kramp_A_to_B:3,kfrozen_A_B:1,kfrozen_A_B:2,kfrozen_A_B:3,kfrozen_A_to_B:1,kfrozen_A_to_B:2,kfrozen_A_to_B:3"

case $SLURM_ARRAY_TASK_ID in
  0) export DRIFT_T1=10 DRIFT_T2=20; OUT=$BASE/t1_10_t2_20 ;;
  1) export DRIFT_T1=1  DRIFT_T2=20; OUT=$BASE/t1_1_t2_20 ;;
esac
mkdir -p "$OUT"

echo "=== [$SLURM_ARRAY_TASK_ID] t1=$DRIFT_T1 t2=$DRIFT_T2 -> $OUT  $(date) ==="
"$PY" run_6scenario_zscores.py --output-dir "$OUT" --only "$DRIFT_ONLY" --n-cores 4 --jobs 3
echo "=== [$SLURM_ARRAY_TASK_ID] DONE  $(date) ==="
