#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics-himem
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem 200GB
#SBATCH -t 4:00:00
#SBATCH --job-name=pca5k_himem
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/pca5k_himem_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/pca5k_himem_%j.err
# One job, all datasets in parallel (xargs -P), then a single 'combine' for the all-dataset elbow figure.
# Already done locally: HS054_Sot48h, PeTrace_E7.5-R1 (PeTrace_* are in the petrace arrays).
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
export PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3 LOG=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
DS=(FM01 FM03 FM05 FM06 FM07 FM08 Watermelon_naive Watermelon_lag Watermelon_late hPSC_20260927 hPSC_endo_yscher hPSC_undifferentiated hPSC_Novo_Nordisk LARRY Michaels_et_al CellTag3 SpaceBar DTT_mouse)
echo "[$(date)] start ${#DS[@]} datasets"
printf '%s\n' "${DS[@]}" | xargs -P ${#DS[@]} -I{} sh -c '$PY pca5k.py --no-combine {} > $LOG/pca5k_{}.out 2> $LOG/pca5k_{}.err; rc=$?; echo "{} exit $rc"; exit $rc'
rc=$?
$PY pca5k.py combine
echo "[$(date)] done (xargs exit $rc; nonzero = at least one dataset failed, see logs/pca5k_<dataset>.err)"
exit $rc
