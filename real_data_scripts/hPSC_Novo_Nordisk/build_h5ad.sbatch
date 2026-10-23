#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -J build_h5ad
#SBATCH -o /home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/build_h5ad.log
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python
D=/home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_Novo_Nordisk
OUT=/home/gzu5140/TwINFER_KA/real_data/Novo_nordisk_hPSC_endo.h5ad
$PY -I $D/build_h5ad.py $OUT && $PY -I $D/validate_h5ad.py $OUT
