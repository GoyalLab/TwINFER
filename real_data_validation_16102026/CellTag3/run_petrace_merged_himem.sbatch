#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics-himem
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 640GB
#SBATCH -t 8:00:00
#SBATCH --job-name=petrace_merged
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/petrace_merged_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/petrace_merged_%j.err
# PeTrace with all replicates (R1+R2+R3) of a day merged: 5k-HVG PCA (50 PCs), then clonal-vs-random pair distance per day.
# E7.5 merged was already run locally. Merged E9.5 = ~1.07M cells and >2^31 nonzeros (int64 indices; tested at small scale):
# memory (estimate, ~250 GB for E9.5 alone + E9.0 ~60 GB running at the same time) has NOT been measured -> check MaxRSS afterwards.
# Stages run as separate processes, so one failing does not stop the others. Outputs are written atomically.
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
export PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3 LOG=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
ST=(PeTrace_E8.0 PeTrace_E8.5 PeTrace_E9.0 PeTrace_E9.5)
echo "[$(date)] PCA start"
printf '%s\n' "${ST[@]}" | xargs -P 4 -I{} sh -c '$PY pca5k.py --no-combine {} > $LOG/pca5k_{}.out 2> $LOG/pca5k_{}.err; rc=$?; echo "{} pca exit $rc"; exit $rc'
rc1=$?
echo "[$(date)] pair distance start"
printf '%s\n' "${ST[@]}" | xargs -P 4 -I{} sh -c '$PY pair_distance.py --no-combine {} > $LOG/pairdist_{}.out 2> $LOG/pairdist_{}.err; rc=$?; echo "{} pairdist exit $rc"; exit $rc'
rc2=$?
$PY pca5k.py combine
$PY pair_distance.py combine
echo "[$(date)] done (pca exit $rc1, pair-distance exit $rc2; nonzero = a stage failed, see logs/pca5k_<stage>.err / pairdist_<stage>.err)"
exit $((rc1 + rc2))
