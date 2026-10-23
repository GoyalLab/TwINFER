#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J export_qs
#SBATCH -o /home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export_seurat.log
/home/gzu5140/.conda/envs/seurat_qs/bin/Rscript /home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_Novo_Nordisk/export_seurat.R
