#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --mem=6GB
#SBATCH --time=4:00:00
#SBATCH --job-name=drift_simulation
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/drift_simulations/logs/slurmLog-%A_%a-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/drift_simulations/logs/slurmLog-%A_%a-%x.err
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

eval "$(conda shell.bash hook)"
conda activate twinfer-code
echo 1
# ~/.conda/envs/twinfer-code/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/drift_multiple_state/simulate_drift_multiple_states.py   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] "${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/drift_multiple_state/simulate_drift_multiple_states.py
"${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state/simulate_drift_multiple_states.py