#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --job-name=motif_probe
#SBATCH -o /home/gzu5140/TwINFER_KA/simulation_data/figure_4/1k_per_motif/logs/probe_%j.out
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate twinfer
# export PYTHONPATH=/home/gzu5140/TwINFER_KA/code/TwINFER/package:$PYTHONPATH   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] export PYTHONPATH=${TWINFER_PROJECT_ROOT}/code/TwINFER/package:$PYTHONPATH
export PYTHONPATH=${TWINFER_CODE_ROOT}/package:$PYTHONPATH
# cd /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/triplet_motif_discrimination   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/triplet_motif_discrimination   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/triplet_motif_discrimination"
echo "START $(date +%s)"
time /home/gzu5140/.conda/envs/twinfer/bin/python simulate_1k.py --config_index 0 --n_reps 1 --rep_offset 9999 --cores 8
echo "END $(date +%s)"
