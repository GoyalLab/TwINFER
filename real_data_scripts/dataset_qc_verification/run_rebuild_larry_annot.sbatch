#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=larry_annot
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_larry_annot_%j.out
# Rerun larry_raw_annotate.py (unmodified) with SOURCE = symlinks to finalized_data/LARRY_data/filtered, so it writes to /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/rebuild/LARRY_annotation/annotated;
# then compare the h5ad element-wise and the record json with finalized_data/LARRY_data/annotated.
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER; source env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
SOURCE=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/rebuild/LARRY_annotation/filtered KNN=15 $PY paper_analysis/larry_hematopoiesis_validation/preprocessing/larry_raw_annotate.py
$PY real_data_scripts/dataset_qc_verification/h5ad_identity.py /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/rebuild/LARRY_annotation/annotated/filtered_annotated_k15.h5ad /gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/LARRY_data/annotated/filtered_annotated_k15.h5ad /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/rebuild/LARRY_annotation/identity.json
cmp /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/rebuild/LARRY_annotation/annotated/annotation_transfer_filtered_k15.json /gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/LARRY_data/annotated/annotation_transfer_filtered_k15.json && echo "record json byte-identical" || diff <(python3 -m json.tool /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/rebuild/LARRY_annotation/annotated/annotation_transfer_filtered_k15.json) <(python3 -m json.tool /gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/LARRY_data/annotated/annotation_transfer_filtered_k15.json) | head -40
