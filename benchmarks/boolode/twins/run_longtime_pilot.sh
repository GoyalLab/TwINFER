#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --mem 8GB
#SBATCH -t 6:00:00
#SBATCH --array=0-6
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins/logs/longtime_%A_%a.out
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/benchmarks/boolode/twins/logs/longtime_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins/logs/longtime_%A_%a.err
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/benchmarks/boolode/twins/logs/longtime_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BOOLODE_PATH:?source clean_code/env.sh first (full BoolODE install: BoolODE/ package + data/)}"

# One-off diagnostic: 1 replicate per network at 10x the production simulation_time, to see
# whether twins diverge more (and whether S vs C decorrelate) once given much longer post-branch
# time than the production runs use. n_pairs kept small (150) since this is exploratory, not a
# production replicate. Previously run locally on the login/shared node and killed -- moved to
# SLURM (one array task per network, 12 cores each) so it doesn't compete with other work.
# SCRIPT_DIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_CODE_ROOT}/benchmarks/boolode/twins
# PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python3}"
# OUT=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/boolode_sims_replicates_longtime   # [2026-09-30 replaced by env.sh variable]
OUT=${TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims_replicates_longtime
N_PAIRS=150

NETWORKS=(B_cell_activation EMT_real Pluripotent_real GSD HSC mCAD VSC)
declare -A T_LONG=([B_cell_activation]=80 [EMT_real]=160 [Pluripotent_real]=160 [GSD]=80 [HSC]=80 [mCAD]=50 [VSC]=50)
declare -A CUSTOM=(
    [B_cell_activation]="custom_networks/B_cell_activation_rules.txt"
    [EMT_real]="custom_networks/EMT_real_rules.txt"
    [Pluripotent_real]="custom_networks/Pluripotent_real_rules.txt"
)

NET=${NETWORKS[$SLURM_ARRAY_TASK_ID]}
T=${T_LONG[$NET]}
EXTRA_ARGS=()
[ -n "${CUSTOM[$NET]:-}" ] && EXTRA_ARGS=(--model-file "${SCRIPT_DIR}/${CUSTOM[$NET]}")

echo "=== ${NET} (T=${T}, 10x production) ==="
"$PYTHON" "${SCRIPT_DIR}/twin_similarity_sweep.py" \
    --network "$NET" --simulation-time "$T" \
    --n-pairs "$N_PAIRS" --workers 12 --seed-offset 0 \
    --output-dir "$OUT" "${EXTRA_ARGS[@]}"
echo "Done: ${NET}"
