#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH --job-name=bcl-convert
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH --mail-user=hanxiao.sun@northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/logs/bcl-convert.%j.out
#SBATCH --error=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/logs/bcl-convert.%j.err
#SBATCH --verbose

module purge
module load bcl-convert

SAMPLE_SHEET_PATH=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/20260924_TwINFER_Diff_bclconvert.csv
OUTPUT_DIR=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/fastq/
FILEPATH=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/260924_VH01990_86_AAF3N2THV/


cd ${FILEPATH}
mkdir -p "${OUTPUT_DIR}"


bcl-convert --bcl-input-directory ${FILEPATH} \
--output-directory ${OUTPUT_DIR} \
--sample-sheet ${SAMPLE_SHEET_PATH} \
--force --bcl-sampleproject-subdirectories true