#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
#SBATCH --time=2:00:00
#SBATCH --job-name=flow_cw
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/flowchart_clone_weighted/logs/%x_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/flowchart_clone_weighted/logs/%x_%j.err
# 4 simulations (A_B, A_to_B, A_B two states, A_to_B two states) x 100 replicates, t1=1 h, t2=20 h; ~25 s per task single-core (tested on A_to_B rep 2).
# Env python called directly (no conda activation in batch). Output on shared storage, never /tmp.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
R=/gpfs/projects/b1255/hzhang/TwINFER_KA
source $R/code/TwINFER/env.sh
cd $R/code/TwINFER/paper_analysis/causal_direction_inference/twinscore_stage_checks
/home/gzu5140/.conda/envs/twinfer-code/bin/python scenario_flowchart_clone_weighted.py \
  --output-dir $R/analysis_data/paper_analysis/causal_direction_inference/flowchart_clone_weighted \
  --n-reps 100 --jobs 24 --n-cores 1
