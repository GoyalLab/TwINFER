#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 24GB
#SBATCH -t 1:00:00
#SBATCH --job-name=petrace_e95_sis
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/petrace_e95_sisters_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/petrace_e95_sisters_%j.err
# PeTrace E9.5 (R1+R2+R3 merged, 1.07M cells): sister-pair vs random-pair PCA distance (all 50 PCs; raw + normalized plots), then combine.
# E7.5, E8.0, E8.5, E9.0 already done locally (peak 0.4-1.0 GB for 4.6k-255k cells -> ~4-5 GB expected for E9.5; 24 GB requested, NOT measured).
# One dataset, so no concurrent combine race; outputs are written atomically.
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
$PY pair_distance.py --no-combine PeTrace_E9.5 || exit 1
$PY pair_distance.py combine
