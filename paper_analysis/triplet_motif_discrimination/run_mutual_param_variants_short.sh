#!/bin/bash
# Same 24 simulations as run_mutual_param_variants.sh (8 variants x 3 replicates), for the `short` partition (max 4 h), account p32655.
# 6 array tasks: task t -> replicate t%3, variant group t//3 (A: control n3 n4 kprod2x | B: mrnahl2x kadd12 koff17 combo).
# Each task runs 4 variants at once on one node (4 workers x 12 cores = 48, +2 spare) so one round fits in the 4 h limit
# (earlier Mutual_regulation replicates took about 1.2-2.5 h in this layout; uncertain). Results go to the same folder as the original script.
# Run `source code/TwINFER/env.sh` before sbatch.
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --job-name=mutvar_s
#SBATCH --array=0-5
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/mutual_param_variants/logs/short-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/mutual_param_variants/logs/short-%A_%a.err
set -eo pipefail
: "${TWINFER_CODE_ROOT:?source code/TwINFER/env.sh before sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=${TWINFER_CODE_ROOT}/package:${PYTHONPATH:-}
OUT=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/mutual_param_variants
REP=$(( SLURM_ARRAY_TASK_ID % 3 ))
if [ $(( SLURM_ARRAY_TASK_ID / 3 )) -eq 0 ]; then V="control n3 n4 kprod2x"; else V="mrnahl2x kadd12 koff17 combo"; fi
~/.conda/envs/twinfer-code/bin/python ${TWINFER_CODE_ROOT}/paper_analysis/triplet_motif_discrimination/run_mutual_param_variants.py \
    --rep ${REP} --out ${OUT} --cores-per 12 --only ${V}
