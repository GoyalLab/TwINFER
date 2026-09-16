#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem 32GB
#SBATCH -t 0:30:00
#SBATCH --job-name=larry_scale_check
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/scale_check_%j.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/scale_check_%j.err
set -euo pipefail

FASTQ=/scratch/gzu5140/ka_twinfer/larry_dataset/LARRY_sorted_and_filtered_barcodes.fastq.gz

echo "[$(date)] Counting distinct (lib,cell,umi,barcode) combos at various NREADS floors"
zcat "$FASTQ" | gawk '
{
  if (substr($0,1,1) == ">") {
    n = split(substr($0,2), h, ",")
    if (n == 3) { lib=h[1]; cell=h[2]; umi=h[3]; want_seq=1 } else { want_seq=0 }
    next
  }
  if (want_seq && $0 != "") {
    count[lib "\t" cell "\t" umi "\t" $0]++
    bc_seen[$0] = 1
    want_seq = 0
    next
  }
  want_seq = 0
}
END {
  total1=0; total5=0; total10=0
  for (k in count) {
    total1++
    if (count[k] >= 5) total5++
    if (count[k] >= 10) total10++
  }
  n_bc = 0
  for (b in bc_seen) n_bc++
  print "distinct combos, NREADS>=1: " total1
  print "distinct combos, NREADS>=5: " total5
  print "distinct combos, NREADS>=10: " total10
  print "distinct raw barcode sequences seen at all (any read count): " n_bc
}'
echo "[$(date)] Done"
