#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_larry
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_larry_%j.out
# stage 1 (zips + barcode fastq -> combined matrix) and stage 2 (finalized combined matrix -> filtered) in parallel; each diffed vs finalized_data
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
L=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs
for ds in LARRY_stage1 LARRY_stage2; do
  $PY rebuild_and_diff.py $ds > $L/rebuild_${ds}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
grep -h "ALL_IDENTICAL" $L/rebuild_LARRY_*_${SLURM_JOB_ID}.log
