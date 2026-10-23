#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J cell_checks
#SBATCH -o /home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/cell_checks.log
D=/home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_Novo_Nordisk
/home/gzu5140/.conda/envs/seurat_qs/bin/Rscript $D/cell_checks.R && \
/home/gzu5140/.conda/envs/twinfer-code/bin/python -I $D/validate_cells_vs_R.py /home/gzu5140/TwINFER_KA/real_data/Novo_nordisk_hPSC_endo.h5ad
