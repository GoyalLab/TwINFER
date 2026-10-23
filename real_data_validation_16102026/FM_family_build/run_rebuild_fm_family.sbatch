#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_fm
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_fm_%j.out
# one process per dataset, in parallel; each reruns the original build script into analysis_data/.../rebuild/<ds>/ and diffs vs finalized_data
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
for ds in FM01 FM06 FM08 FM05 FM07 HS054_Sot48h Watermelon; do
  $PY rebuild_and_diff.py $ds > /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_${ds}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
grep -h "ALL_IDENTICAL" /gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_*_${SLURM_JOB_ID}.log
