#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 1 -c 128 --mem=0
#SBATCH --job-name=burn12k
#SBATCH --array=0-5
#SBATCH -o /gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/burnin12000/logs/b12k_%A_%a.out   # [2026-10-01 was /scratch/gzu5140/motif_burnin12000/logs; the logs dir must exist before sbatch: mkdir -p it first]
set -e
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# export PYTHONPATH=/home/gzu5140/TwINFER_KA/code/TwINFER/package:${PYTHONPATH:-}   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] export PYTHONPATH=${TWINFER_PROJECT_ROOT}/code/TwINFER/package:${PYTHONPATH:-}
export PYTHONPATH=${TWINFER_CODE_ROOT}/package:${PYTHONPATH:-}
# cd /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/triplet_motif_discrimination   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/triplet_motif_discrimination   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/triplet_motif_discrimination"
CFG=$(( SLURM_ARRAY_TASK_ID / 2 ))
REP=$(( SLURM_ARRAY_TASK_ID % 2 ))
echo "task=$SLURM_ARRAY_TASK_ID cfg=$CFG rep=$REP host=$(hostname) cores=$SLURM_CPUS_PER_TASK"
/home/gzu5140/.conda/envs/twinfer/bin/python simulate_burnin12000.py \
    --config_index "$CFG" --n_reps 1 --rep_offset "$REP" --cores 128
