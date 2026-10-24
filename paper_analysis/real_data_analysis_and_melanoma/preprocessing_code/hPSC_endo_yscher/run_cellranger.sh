#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --job-name=cellranger
#SBATCH --array=0-2
# hPSC endoderm RNA: Cell Ranger count of the three samples A2, endo_T0, endo_T1 (array 0-2) from the bcl-convert FASTQs. NOT rerun for Figure 5 (its output is used as given).
# Paths come from the raw_data links; the 10x reference is not part of the project and must be given:  CELLRANGER_REF=/path/to/refdata-gex-GRCh38-2024-A sbatch run_cellranger.sh
# (the original run used the 10x GRCh38-2024-A reference in another user's home folder, which is not readable from here).
# Submit with  sbatch -o <log folder>/cellranger_%A_%a.out -e <log folder>/cellranger_%A_%a.err ...
module purge
module load cellranger

set -euo pipefail

ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
RAW=${FIG5_RAW_DATA:-$ROOT/final_figure_data/real_data_analysis_and_melanoma/raw_data}
HANXIAO=$RAW/hPSC_endo_cellranger_and_fastq_Hanxiao
FASTQ_ROOT=$HANXIAO/fastq
OUT_DIR=$HANXIAO/cellranger_out
REF=${CELLRANGER_REF:?set CELLRANGER_REF to the 10x refdata-gex-GRCh38-2024-A folder}
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
