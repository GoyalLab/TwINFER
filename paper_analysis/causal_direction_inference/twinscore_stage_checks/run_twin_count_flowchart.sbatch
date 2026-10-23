#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
#SBATCH --time=3:00:00
#SBATCH --job-name=twin_count_flow
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/twin_count_power/flowchart_singletons/logs/%x_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/twin_count_power/flowchart_singletons/logs/%x_%j.err
# Flowchart (Stage I step1_z, Stage II z_het, Stage III z_d) vs number of twins N = 250,376,500,712,1000,1500 on the 4 two-gene simulations x 100 replicates (400 files),
# design = N twins at t1 + N other twins at t2 + 3,000 cross (singleton) cells per sample, 5,000 null draws. ~2.5 min per file single-core (smoke test: ~1 min for N = 250 and 1500) -> ~45 min on 24 cores.
# Resumable: one CSV per file in the output folder, existing files are skipped; resubmit if the wall time runs out. Env python called directly (no conda activation); output on shared storage.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
R=/gpfs/projects/b1255/hzhang/TwINFER_KA
source $R/code/TwINFER/env.sh
cd $R/code/TwINFER/paper_analysis/causal_direction_inference/twinscore_stage_checks
/home/gzu5140/.conda/envs/twinfer-code/bin/python twin_count_flowchart.py \
  $R/analysis_data/paper_analysis/causal_direction_inference/twin_count_power/flowchart_singletons \
  --nreps 100 --jobs 24 --shuffles 5000
