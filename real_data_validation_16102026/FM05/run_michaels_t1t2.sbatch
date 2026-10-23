#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16GB
#SBATCH -t 0:30:00
#SBATCH --job-name=michaels_t1t2
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/michaels_t1t2_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/michaels_t1t2_%j.err
# Michaels et al. T1/T2 (the part WITH clone labels): 5k-HVG PCA (50 PCs), then clonal-vs-random pair distance. QC verification already done locally.
# Login node kills the PCA at ~2.9 GB (limit 2.8 GB); measured 2.9 GB peak, so 16 GB is ample.
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
$PY pca5k.py --no-combine Michaels_T1T2 || exit 1
$PY pair_distance.py --no-combine Michaels_T1T2 || exit 1
$PY pca5k.py combine
$PY pair_distance.py combine
