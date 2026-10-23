#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_novo
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_novo_export_%j.out
# Rerun export_seurat.R (only the output folder changed) on the authors' Seurat .qs object, then byte-compare every exported file
# with analysis_data/hPSC_Novo_Nordisk/export (the input of build_qc_per_sample.py).
/home/gzu5140/.conda/envs/seurat_qs/bin/Rscript /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/rebuild/hPSC_Novo_Nordisk/code/export_seurat_rerun.R
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/rebuild/hPSC_Novo_Nordisk/export_rerun; for f in *; do cmp -s "$f" "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export/$f" && echo "$f byte-identical" || echo "$f DIFFERS"; done
