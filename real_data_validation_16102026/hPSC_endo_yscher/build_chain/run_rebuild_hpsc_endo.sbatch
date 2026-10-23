#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_endo
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_endo_%j.out
# hPSC endo (yscher's index_fix/B build), rebuilt with yscher's unmodified scripts and environments, outputs to our scratch:
#  A  build_endo_dataset.py on yscher's stepThree tables  -> h5ad, compared element-wise with finalized_data copy
#  B1 md5(cat L001 L002) == md5(yscher's concatenated FASTQ)  (R1 and R2)
#  B2 Hanxiao's barcode pipeline stepOne -> stepTwo -> stepThree (variant B: shipped staggers, --index-mismatches 2) from the FASTQs
#  B3 stepThree tables compared byte-wise with yscher's; build_endo_dataset.py on OUR stepThree -> h5ad compared with finalized copy
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA; W=$ROOT/analysis_data/dataset_qc_verification/rebuild/hPSC_endo_yscher; mkdir -p $W/A $W/B
YR="/gpfs/projects/b1255/yscher/Transcriptomic Distance"; YB="$YR/datasets/twinfer_diff_endo/index_fix/B"
YPY=/hpc/software/mamba/24.3.0/bin/python3; CMP="/home/gzu5140/.conda/envs/twinfer-code/bin/python3 $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification/h5ad_identity.py"
FINH=$ROOT/finalized_data/hPSC_endo_yscher_data/endo_kbnac_spliced_unspliced.h5ad
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
( ENDO_S3="$YB/stepThree" ENDO_OUT=$W/A $YPY "$YR/helpers/build_endo_dataset.py" > $W/A/build.log 2>&1
  $CMP $W/A/endo_kbnac_spliced_unspliced.h5ad $FINH $W/A/identity_vs_finalized.json > $W/A/identity.log 2>&1
  $CMP $W/A/endo_counts_mito20_min200genes.h5ad "$YB/endo_counts_mito20_min200genes.h5ad" $W/A/identity_counts_vs_hers.json > $W/A/identity_counts.log 2>&1 ) &
H=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/fastq; T=/projects/b1042/GoyalLab/Yuval/endo_index_fix_tmp
( for r in R1 R2; do a=$(cat $H/Undetermined_S0_L001_${r}_001.fastq.gz $H/Undetermined_S0_L002_${r}_001.fastq.gz | md5sum | cut -d' ' -f1); b=$(md5sum < $T/Undetermined_combined_${r}.fastq.gz | cut -d' ' -f1); echo "$r lanes_cat=$a combined=$b identical=$([ $a = $b ] && echo yes || echo NO)"; done > $W/B/fastq_concat_md5.txt ) &
E=/projects/b1042/GoyalLab/Yuval/envs/fatemapbc; HX=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff
S1=$HX/scripts/Updated_10XBarcodeMatching/SubmissionScripts/barcode; S3=/projects/b1042/GoyalLab/Hanxiao/TwINFER_2026scRNAseq/scripts/Updated_10XBarcodeMatching/SubmissionScripts/barcode
STAG="$YR/datasets/twinfer_diff_endo/index_fix/staggers_B_sheet_index.txt"
( $E/bin/python $S1/stepOne.py --r1 $T/Undetermined_combined_R1.fastq.gz --r2 $T/Undetermined_combined_R2.fastq.gz --staggers "$STAG" --outdir $W/B/stepOne/ --mode GFP --index-mismatches 2 > $W/B/barcode.log 2>&1
  $E/bin/Rscript $S1/stepTwo.R "$STAG" "" $W/B/stepOne/ $W/B/stepTwo/ endo_T0 endo_T15 >> $W/B/barcode.log 2>&1
  $E/bin/python $S3/stepThree.py $W/B/stepTwo/ $W/B/stepThree/ $HX/scripts/starcode/starcode endo_T0 endo_T15 >> $W/B/barcode.log 2>&1
  for s in endo_T0 endo_T15; do cmp $W/B/stepThree/$s/stepThreeStarcodeShavedReads.txt "$YB/stepThree/$s/stepThreeStarcodeShavedReads.txt" && echo "$s stepThree byte-identical" || echo "$s stepThree DIFFERS"; done > $W/B/stepThree_cmp.txt
  mkdir -p $W/B/build; ENDO_S3=$W/B/stepThree ENDO_OUT=$W/B/build $YPY "$YR/helpers/build_endo_dataset.py" > $W/B/build/build.log 2>&1
  $CMP $W/B/build/endo_kbnac_spliced_unspliced.h5ad $FINH $W/B/build/identity_vs_finalized.json > $W/B/build/identity.log 2>&1 ) &
wait
echo "== A"; tail -3 $W/A/identity.log; tail -3 $W/A/identity_counts.log; echo "== B"; cat $W/B/fastq_concat_md5.txt $W/B/stepThree_cmp.txt; tail -3 $W/B/build/identity.log
