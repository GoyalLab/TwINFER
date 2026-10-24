#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_endo
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# Figure 5 preprocessing, hPSC endoderm (yscher's index_fix/B build) rebuilt with yscher's scripts (copied here) and her environments. Output: <REBUILD>/hPSC_endo_yscher/{A,B}:
#  A  build_endo_dataset.py on yscher's stepThree tables -> h5ad, compared element-wise with final_data/hPSC_endo_yscher_data
#  B1 md5(cat L001 L002) == md5(yscher's concatenated FASTQ)  (R1 and R2)
#  B2 Hanxiao's barcode pipeline stepOne -> stepTwo -> stepThree (variant B: shipped staggers, --index-mismatches 2) from the FASTQs
#  B3 stepThree tables compared byte-wise with yscher's; build_endo_dataset.py on OUR stepThree -> h5ad compared with the final file
# Not rerun: bcl-convert and Cell Ranger (bclconvert_20260924_TwINFER_Diff.sh, run_cellranger.sh) and kb count.   Usage: sbatch run_rebuild_hpsc_endo.sbatch
set -uo pipefail
ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
FIG=$ROOT/final_figure_data/real_data_analysis_and_melanoma
RAW=${FIG5_RAW_DATA:-$FIG/raw_data}                                  # links to the raw data (see raw_data/)
FINAL=$FIG/final_data                                                # links to the final objects
REBUILD=${FIG5_REBUILD_DIR:-$FIG/analysis_data/preprocessing_rebuild}  # rebuilt files go here, never into final_data
CODE=$ROOT/code/TwINFER/paper_analysis/real_data_analysis_and_melanoma/preprocessing_code
PY=${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}   # conda activate does not work in batch jobs: call the environment python directly
export TWINFER_PROJECT_ROOT=$ROOT
W=$REBUILD/hPSC_endo_yscher; mkdir -p "$W/A" "$W/B"
YB=$RAW/yscher_twinfer_diff_endo/index_fix/B                            # her variant-B outputs (raw_data link)
YPY=$RAW/external_tools/mamba_python3.10; CMP="$PY $CODE/h5ad_identity.py"
BUILD=$CODE/hPSC_endo_yscher/build_endo_dataset.py
FINH=$FINAL/hPSC_endo_yscher_data/endo_kbnac_spliced_unspliced.h5ad
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
( ENDO_S3="$YB/stepThree" ENDO_OUT=$W/A $YPY "$BUILD" > $W/A/build.log 2>&1
  $CMP $W/A/endo_kbnac_spliced_unspliced.h5ad $FINH $W/A/identity_vs_finalized.json > $W/A/identity.log 2>&1
  $CMP $W/A/endo_counts_mito20_min200genes.h5ad "$YB/endo_counts_mito20_min200genes.h5ad" $W/A/identity_counts_vs_hers.json > $W/A/identity_counts.log 2>&1 ) &
H=$RAW/hPSC_endo_cellranger_and_fastq_Hanxiao/fastq; T=$RAW/hPSC_endo_combined_fastq_yscher       # lane FASTQs (Hanxiao) and yscher's concatenated FASTQs
( for r in R1 R2; do a=$(cat $H/Undetermined_S0_L001_${r}_001.fastq.gz $H/Undetermined_S0_L002_${r}_001.fastq.gz | md5sum | cut -d' ' -f1); b=$(md5sum < $T/Undetermined_combined_${r}.fastq.gz | cut -d' ' -f1); echo "$r lanes_cat=$a combined=$b identical=$([ $a = $b ] && echo yes || echo NO)"; done > $W/B/fastq_concat_md5.txt ) &
E=$RAW/external_tools/fatemapbc_env                                                              # yscher's conda environment with python, R and the barcode tools
S1=$CODE/hPSC_endo_yscher/barcode_pipeline_twinfer_diff; S3=$CODE/hPSC_endo_yscher/barcode_pipeline_TwINFER_2026      # copies of Hanxiao's stepOne.py / stepTwo.R / stepThree.py
STARCODE=$RAW/external_tools/starcode/starcode
# the stagger sheet lists the Cell Ranger folders of Hanxiao in its 4th column: written to <REBUILD> with those paths pointed at the raw_data link (the rest of the file is unchanged)
STAG=$W/B/staggers_B_sheet_index.txt; sed "s#/projects/b1042/GoyalLab/Hanxiao/twinfer_diff#$RAW/hPSC_endo_cellranger_and_fastq_Hanxiao#" "$CODE/hPSC_endo_yscher/staggers_B_sheet_index.txt" > "$STAG"
( $E/bin/python $S1/stepOne.py --r1 $T/Undetermined_combined_R1.fastq.gz --r2 $T/Undetermined_combined_R2.fastq.gz --staggers "$STAG" --outdir $W/B/stepOne/ --mode GFP --index-mismatches 2 > $W/B/barcode.log 2>&1
  $E/bin/Rscript $S1/stepTwo.R "$STAG" "" $W/B/stepOne/ $W/B/stepTwo/ endo_T0 endo_T15 >> $W/B/barcode.log 2>&1
  $E/bin/python $S3/stepThree.py $W/B/stepTwo/ $W/B/stepThree/ $STARCODE endo_T0 endo_T15 >> $W/B/barcode.log 2>&1
  for s in endo_T0 endo_T15; do cmp $W/B/stepThree/$s/stepThreeStarcodeShavedReads.txt "$YB/stepThree/$s/stepThreeStarcodeShavedReads.txt" && echo "$s stepThree byte-identical" || echo "$s stepThree DIFFERS"; done > $W/B/stepThree_cmp.txt
  mkdir -p $W/B/build; ENDO_S3=$W/B/stepThree ENDO_OUT=$W/B/build $YPY "$BUILD" > $W/B/build/build.log 2>&1
  $CMP $W/B/build/endo_kbnac_spliced_unspliced.h5ad $FINH $W/B/build/identity_vs_finalized.json > $W/B/build/identity.log 2>&1 ) &
wait
echo "== A"; tail -3 $W/A/identity.log; tail -3 $W/A/identity_counts.log; echo "== B"; cat $W/B/fastq_concat_md5.txt $W/B/stepThree_cmp.txt; tail -3 $W/B/build/identity.log
