#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH --job-name=bcl-convert
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH --verbose
# hPSC endoderm RNA: bcl-convert of the 2026-09-24 sequencing run (TwINFER_Diff) into FASTQs. NOT rerun for Figure 5 (the FASTQs are used as given).
# Paths come from the raw_data link hPSC_endo_cellranger_and_fastq_Hanxiao (Hanxiao's twinfer_diff folder); the sample sheet is 20260924_TwINFER_Diff_bclconvert.csv in this folder.
# Submit with  sbatch -o <log folder>/bcl-convert.%j.out -e <log folder>/bcl-convert.%j.err bclconvert_20260924_TwINFER_Diff.sh

module purge
module load bcl-convert

ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
RAW=${FIG5_RAW_DATA:-$ROOT/final_figure_data/real_data_analysis_and_melanoma/raw_data}
HANXIAO=$RAW/hPSC_endo_cellranger_and_fastq_Hanxiao
SAMPLE_SHEET_PATH=$ROOT/code/TwINFER/paper_analysis/real_data_analysis_and_melanoma/preprocessing_code/hPSC_endo_yscher/20260924_TwINFER_Diff_bclconvert.csv
OUTPUT_DIR=$HANXIAO/fastq/
FILEPATH=$HANXIAO/260924_VH01990_86_AAF3N2THV/


cd ${FILEPATH}
mkdir -p "${OUTPUT_DIR}"


bcl-convert --bcl-input-directory ${FILEPATH} \
--output-directory ${OUTPUT_DIR} \
--sample-sheet ${SAMPLE_SHEET_PATH} \
--force --bcl-sampleproject-subdirectories true
