#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_dtt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_dtt_%j.out
bash /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification/dtt_rebuild/run_dtt_rebuild.sh
