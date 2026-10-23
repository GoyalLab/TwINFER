#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=top30_xcorr
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/stress_tf_targets/top30_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/stress_tf_targets/top30_%j.out
# 30 most expressed TFs vs their CollecTRI targets, cross-correlation across shared clones, both directions, 3 timepoint pairs.
export OMP_NUM_THREADS=2
/home/gzu5140/.conda/envs/twinfer/bin/python -I /home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_undifferentiated/07_top30_tf_collectri_crosscorr.py
