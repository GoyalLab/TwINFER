#!/bin/bash
# 8 parameter variants of Mutual_regulation x 3 replicates = 24 simulations. One array task per replicate (0,1,2);
# each task runs all 8 variants at once on one node (8 workers x 6 cores = 48, +2 spare cores). 6000 h burn-in, 48 h twins, 6000 cells.
# Variants: control | n3 | n4 | kprod2x | mrnahl2x | kadd12 | koff17 | combo (n=4, k_off=17.2, k_add=12). K is recomputed from the steady state.
# Run `source code/TwINFER/env.sh` before sbatch.
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=60GB
#SBATCH --time=12:00:00
#SBATCH --job-name=mutvar
#SBATCH --array=0-2
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/mutual_param_variants/logs/slurm-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/mutual_param_variants/logs/slurm-%A_%a.err
set -eo pipefail
: "${TWINFER_CODE_ROOT:?source code/TwINFER/env.sh before sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=${TWINFER_CODE_ROOT}/package:${PYTHONPATH:-}
OUT=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/mutual_param_variants
~/.conda/envs/twinfer-code/bin/python ${TWINFER_CODE_ROOT}/paper_analysis/triplet_motif_discrimination/run_mutual_param_variants.py \
    --rep ${SLURM_ARRAY_TASK_ID} --out ${OUT} --cores-per 6
