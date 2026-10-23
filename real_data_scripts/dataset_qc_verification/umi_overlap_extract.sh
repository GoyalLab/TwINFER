#!/bin/bash
# Step 2: pull the reads of the target cell barcodes from one library's SRA FASTQ.
# usage: umi_overlap_extract.sh <LIB> <fastq.gz or prefix.fastq.gz> [OUTDIR]
# output: <OUTDIR>/reads_<LIB>.tsv.gz  (cell barcode, UMI, first 40 bp of the read)
LIB=$1; FQ=$2
OUT=${3:-/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/larry_raw_sra/umi_overlap}
TG=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/larry_raw_sra/umi_overlap/targets_$LIB.txt
zcat "$FQ" 2>/dev/null | awk -v OFS="\t" 'FNR==NR{t[$1];next}
  FNR%4==1{split($2,a,":"); keep=(a[1] in t); bc=a[1]; um=a[2]}
  FNR%4==2&&keep{print bc,um,substr($0,1,40)}' "$TG" - | gzip > "$OUT/reads_$LIB.tsv.gz"
echo "$LIB: $(zcat "$OUT/reads_$LIB.tsv.gz" | wc -l) reads kept"
