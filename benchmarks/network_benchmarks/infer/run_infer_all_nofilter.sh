#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=62
#SBATCH --mem 24GB
#SBATCH -t 16:00:00
#SBATCH --job-name=infer_all_nofilter
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs/infer_all_nofilter_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs/infer_all_nofilter_%j.err
set -o pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/logs
# mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter/logs
# mkdir -p /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate_nofilter/logs   # [2026-09-30 replaced by env.sh variable]
mkdir -p ${TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_multistate_nofilter/logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[$(date)] JOB $SLURM_JOB_ID on $(hostname)"
# echo "Python: $(~/.conda/envs/twinfer-code/bin/python -V)"   # [2026-09-30 replaced by env.sh variable]
echo "Python: $("${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}" -V)"
echo

# cd "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis"   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/benchmarks/network_benchmarks/infer"

# ---- Step 1: original 8 real-network sims (GSD/HSC/VSC/mCAD/EMT/Pluripotent/
# B_cell_activation/Circadian_cycle, 75 replicates), Step 1-3 coverage gates
# disabled -- see infer_network_simulation_real_network_nofilter.py's
# docstring for exact parameter changes and rationale.
echo "[$(date)] ===== Step 1/2: original 8 real-network sims (no-filter) ====="
# ~/.conda/envs/twinfer-code/bin/python -u infer_network_simulation_real_network_nofilter.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}" -u infer_network_simulation_real_network_nofilter.py
status1=$?
echo "[$(date)] Step 1/2 finished with exit code $status1"

# ---- Step 2: multistate/seeded real-network sims (GSD/HSC/EMT/VSC/mCAD
# multistate+seeded variants, 24 replicates), same coverage gates disabled --
# see infer_network_simulation_multistate_nofilter.py's docstring. Runs
# regardless of Step 1's outcome -- the two tracks are independent, and a
# problem in one (e.g. a single unexpected task-discovery error) shouldn't
# block the other from completing.
echo
echo "[$(date)] ===== Step 2/2: multistate/seeded real-network sims (no-filter) ====="
# ~/.conda/envs/twinfer-code/bin/python -u infer_network_simulation_multistate_nofilter.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-$HOME/.conda/envs/twinfer-code/bin/python}" -u infer_network_simulation_multistate_nofilter.py
status2=$?
echo "[$(date)] Step 2/2 finished with exit code $status2"

echo
echo "[$(date)] Step 1 exit=$status1, Step 2 exit=$status2"
if [ "$status1" -ne 0 ] || [ "$status2" -ne 0 ]; then
    exit 1
fi
exit 0
