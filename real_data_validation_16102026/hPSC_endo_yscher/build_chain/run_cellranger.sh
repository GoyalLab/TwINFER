#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --job-name=cellranger
#SBATCH --array=0-2
#SBATCH --output=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/cellranger_out/logs/cellranger_%A_%a.out
#SBATCH --error=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/cellranger_out/logs/cellranger_%A_%a.err

module purge
module load cellranger

set -euo pipefail

FASTQ_ROOT=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/fastq
OUT_DIR=/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/cellranger_out
REF=/home/hsv2210/refdata-gex-GRCh38-2024-A
PROJECTS=(endo meso)

samples=(A2 endo_T0 endo_T1)
s=${samples[$SLURM_ARRAY_TASK_ID]}

# Work out which project folder holds this sample's FASTQs
found=()
for p in "${PROJECTS[@]}"; do
  if [[ -n $(find "${FASTQ_ROOT}/${p}" -name "${s}_S*_R1_*.fastq.gz" -print -quit) ]]; then
    found+=("$p")
  fi
done

if [[ ${#found[@]} -eq 0 ]]; then
  echo "ERROR: no FASTQs named ${s}_S*_R1_*.fastq.gz under ${FASTQ_ROOT}/{${PROJECTS[*]}}" >&2
  exit 1
elif [[ ${#found[@]} -gt 1 ]]; then
  echo "ERROR: sample ${s} found in multiple projects: ${found[*]}" >&2
  exit 1
fi
project=${found[0]}
echo "Sample ${s} -> project ${project}"

cd "$OUT_DIR"

cellranger count --id="${s}" \
  --fastqs="${FASTQ_ROOT}" \
  --project="${project}" \
  --sample="${s}" \
  --transcriptome="${REF}" \
  --create-bam=true \
  --localcores=16 \
  --localmem=76