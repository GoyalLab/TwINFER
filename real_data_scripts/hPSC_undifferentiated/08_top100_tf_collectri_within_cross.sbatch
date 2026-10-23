#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=top100_xcorr
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/tf_collectri_top100/top100_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/tf_collectri_top100/top100_%j.out
# Top-100 most expressed TFs: within-sample + cross-clone correlations, |x|-|y| asymmetry, plus 5,000 clone-pairing scrambles (empirical p-values per pair, BH-FDR). Script 08 with NTF=100 NPERM=5000.
export OMP_NUM_THREADS=4 NTF=100 NPERM=5000 OUTDIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/tf_collectri_top100/
/home/gzu5140/.conda/envs/twinfer/bin/python -I /home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_undifferentiated/08_extended_tf_collectri_within_cross.py
