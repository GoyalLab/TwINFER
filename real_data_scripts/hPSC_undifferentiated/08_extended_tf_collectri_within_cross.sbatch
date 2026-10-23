#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=ext_tf_xcorr
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/tf_collectri_extended/ext_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/tf_collectri_extended/ext_%j.out
# All TFs with >=5% cells and >=20 CollecTRI targets: within-sample + cross-clone correlations, |x|-|y| asymmetry.
export OMP_NUM_THREADS=4 NTF=all
/home/gzu5140/.conda/envs/twinfer/bin/python -I /home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_undifferentiated/08_extended_tf_collectri_within_cross.py
