#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=rebuild_fullsheet
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_fullsheet_%j.out
# FM01/FM03/FM06/FM08 rebuilt with paper mito cutoffs and singletCode on the whole sheet; new folders, finalized_data/FM*_data untouched
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER
source env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
OUT=/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/rebuild_fullsheet_2026-10-16
L=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs
for ds in FM01 FM03 FM06 FM08; do $PY -m paper_analysis.fatemap_pipeline.build_fm_fullsheet_rebuild $ds $OUT > $L/rebuild_fullsheet_${ds}_${SLURM_JOB_ID}.log 2>&1 & done
wait
for ds in FM01 FM03 FM06 FM08; do echo $ds; tail -22 $L/rebuild_fullsheet_${ds}_${SLURM_JOB_ID}.log | grep -E "raw=|n_rescued|n_clones"; done
