#!/bin/bash
# 3 motifs x 1000 replicates (Fan_out, Feed_forward, Mutual_regulation; median parameter row 0; 6000 cells; 6000 h burn-in; 48 h twins).
# 15 array tasks; each runs 200 replicates (global index g%15==task) as 4 parallel workers x 12 cores inside one 50-core job (4x12 + 2 spare cores for I/O).
# Resumable: a replicate whose df_rows_*.csv exists is skipped. The 1.7 GB simulation_before_division file is NOT written.
# Run `source code/TwINFER/env.sh` before sbatch (exports TWINFER_CODE_ROOT; sbatch passes the environment).
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=100GB
#SBATCH --time=36:00:00
#SBATCH --job-name=motif1k
#SBATCH --array=0-14
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2/logs/slurmLog-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2/logs/slurmLog-%A_%a.err
set -eo pipefail
: "${TWINFER_CODE_ROOT:?source code/TwINFER/env.sh before sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=${TWINFER_CODE_ROOT}/package:${PYTHONPATH:-}
OUT=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2
~/.conda/envs/twinfer-code/bin/python ${TWINFER_CODE_ROOT}/paper_analysis/triplet_motif_discrimination/run_1k_per_motif_multicore.py \
    --slice ${SLURM_ARRAY_TASK_ID} --n-slices 15 --jobs 4 --cores-per 12 --out ${OUT}
