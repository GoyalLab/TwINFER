#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=fm01_tf_all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm01/tf_vs_all_genes/logs/%x_%j.log
set -eu
R=/gpfs/projects/b1255/hzhang/TwINFER_KA; PY=/home/gzu5140/.conda/envs/twinfer/bin/python
export OMP_NUM_THREADS=4
$PY -I $R/code/TwINFER/real_data_scripts/FM01_twinscore/16_fm01_tf_vs_all_genes.py $R/analysis_data/fm01/tf_vs_all_genes KDM5A,KDM5B
$PY -I $R/code/TwINFER/real_data_scripts/FM01_twinscore/16_fm01_tf_kdm5_table.py $R/analysis_data/fm01/tf_vs_all_genes
