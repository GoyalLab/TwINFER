#!/bin/bash
# Throughput test: how many Mutual_regulation replicates per hour does ONE 50-core node deliver under different layouts?
# One array task per layout, each runs exactly ONE round (workers = replicates), so wall time ~= time per replicate under that layout:
#   task 0: 4 workers x 12 cores (the layout of job 8232091, control)   task 1: 8 workers x 6 cores   task 2: 16 workers x 3 cores
# Output goes to a separate test folder (not the production folder). Compare "ELAPSED_SEC" in the .out logs: replicates/hour = N / ELAPSED_SEC * 3600.
# Run `source code/TwINFER/env.sh` before sbatch.
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=120GB
#SBATCH --time=04:00:00
#SBATCH --job-name=motif1k_layout
#SBATCH --array=0-2
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2_layout_test/logs/layout-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2_layout_test/logs/layout-%A_%a.err
set -eo pipefail
: "${TWINFER_CODE_ROOT:?source code/TwINFER/env.sh before sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=${TWINFER_CODE_ROOT}/package:${PYTHONPATH:-}
OUT=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2_layout_test/task${SLURM_ARRAY_TASK_ID}
case ${SLURM_ARRAY_TASK_ID} in
  0) W=4;  C=12 ;;
  1) W=8;  C=6  ;;
  2) W=16; C=3  ;;
esac
G=$(seq 2000 $((2000 + W - 1)))      # W Mutual_regulation replicates, one per worker (same replicate indices in every layout)
echo "layout workers=$W cores_per=$C replicates=$W node=$(hostname) start=$(date +%s)"
T0=$(date +%s)
~/.conda/envs/twinfer-code/bin/python ${TWINFER_CODE_ROOT}/paper_analysis/triplet_motif_discrimination/run_1k_per_motif_multicore.py \
    --slice 0 --n-slices 1 --jobs ${W} --cores-per ${C} --out ${OUT} --g ${G}
T1=$(date +%s)
echo "ELAPSED_SEC=$((T1 - T0)) workers=$W cores_per=$C replicates=$W"
