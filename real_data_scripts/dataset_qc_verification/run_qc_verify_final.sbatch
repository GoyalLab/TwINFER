#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=qcverify_final
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/qcverify_final_%j.out
# ONE job (no arrays): raw-vs-final QC verification + plots on the UPDATED finalized_data (old verify.json etc. archived in archive_before_fullsheet_downstream_2026-10-16/qc_verify/).
# 3 datasets at a time. FM07, HS054, hPSC/Novo/Michaels/CellTag/SpaceBar/DTT/PeTrace are unchanged and not rerun.
set -uo pipefail
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3; L=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs; J=$SLURM_JOB_ID
echo FM01 FM03 FM05 FM06 FM08 Watermelon LARRY | tr ' ' '\n' | xargs -P 3 -I{} sh -c "$PY qc_verify.py {} > $L/qcverify_final_{}_$J.log 2>&1 && echo OK {} || echo FAILED {}"
echo done
