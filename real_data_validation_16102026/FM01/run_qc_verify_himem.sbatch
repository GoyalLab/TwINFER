#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics-himem
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 160GB
#SBATCH -t 3:00:00
#SBATCH --job-name=qcverify_himem
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/qcverify_himem_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/qcverify_himem_%j.err
# One job, all datasets in parallel (xargs -P). Peak memory = sum of the concurrent datasets (Watermelon ~25 GB, the rest <= ~8 GB).
# Already done locally: HS054_Sot48h, Michaels_et_al, DTT_mouse, PeTrace_E7.5-R1.
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
export PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3 LOG=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
DS=(FM01 FM03 FM05 FM06 FM07 FM08 Watermelon hPSC_20260927 LARRY CellTag3 SpaceBar)
echo "[$(date)] start ${#DS[@]} datasets"
printf '%s\n' "${DS[@]}" | xargs -P ${#DS[@]} -I{} sh -c '$PY qc_verify.py {} > $LOG/qcverify_{}.out 2> $LOG/qcverify_{}.err; rc=$?; echo "{} exit $rc"; exit $rc'
rc=$?
echo "[$(date)] done (xargs exit $rc; nonzero = at least one dataset failed, see logs/qcverify_<dataset>.err)"
exit $rc
