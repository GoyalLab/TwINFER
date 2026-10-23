#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=stress_perm
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/stress_tf_targets/perm_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/stress_tf_targets/perm_%j.out
# 10,000 clone-pairing scrambles, stress TF vs Hotspot-module genes, both directions, T0-T10 / T10-T15 / T0-T15.
export OMP_NUM_THREADS=4 NPERM=10000
/home/gzu5140/.conda/envs/twinfer/bin/python -I /home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_undifferentiated/06_stress_tf_module_crosscorr_perm.py
