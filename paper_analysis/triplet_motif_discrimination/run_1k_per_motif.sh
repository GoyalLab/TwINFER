#!/bin/bash
# 1000 TwINFER simulations for each of 3 motifs, median parameter row (row 0), n_cells=6000.
#   config_index 0 = Fan_out            (Z->A, Z->B)
#   config_index 1 = Feed_forward       (Z->A, Z->B, A->B)
#   config_index 2 = Mutual_regulation  (Z->A, Z->B, A->B, B->A)   <- "regulated mutual"
# One replicate per array task -> 3000 fully independent tasks, maximum parallelism.
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --job-name=motif1k
#SBATCH --array=0-2999%100
#SBATCH -o /gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif/logs/m1k_%A_%a.out   # [2026-10-01 was /scratch/gzu5140/motif_1k/logs; #SBATCH cannot use variables]

set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# export PYTHONPATH=/home/gzu5140/TwINFER_KA/code/TwINFER/package:${PYTHONPATH:-}   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] export PYTHONPATH=${TWINFER_PROJECT_ROOT}/code/TwINFER/package:${PYTHONPATH:-}
export PYTHONPATH=${TWINFER_CODE_ROOT}/package:${PYTHONPATH:-}
# cd /home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/triplet_motif_discrimination   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/triplet_motif_discrimination   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/triplet_motif_discrimination"

CFG=$(( SLURM_ARRAY_TASK_ID / 1000 ))    # 0,1,2 -> motif
REP=$(( SLURM_ARRAY_TASK_ID % 1000 ))    # replicate 0..999

echo "task=$SLURM_ARRAY_TASK_ID config_index=$CFG rep=$REP host=$(hostname) start=$(date +%s)"
/home/gzu5140/.conda/envs/twinfer/bin/python simulate_1k.py \
    --config_index "$CFG" --n_reps 1 --rep_offset "$REP" --cores 8
echo "DONE task=$SLURM_ARRAY_TASK_ID end=$(date +%s)"
