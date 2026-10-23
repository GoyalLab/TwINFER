#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 64GB
#SBATCH -t 2:00:00
#SBATCH --job-name=pairdist_single
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/pairdist_single_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/pairdist_single_%j.err
# Clonal vs random pair PCA distance (all 50 PCs, raw + normalized plots). One job, 8 datasets at a time (xargs -P 8),
# --no-combine in the workers (no shared-file reads while others write), ONE combine at the end.
# Needs the pca5k outputs (all exist). Michaels has no clones; HS054_Sot48h was run locally. PeTrace is merged per day in run_petrace_merged_himem.sbatch.
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA
export PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3 LOG=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification
DS=(FM01 FM03 FM05 FM06 FM07 FM08 Watermelon_naive Watermelon_lag Watermelon_late hPSC_20260927 hPSC_endo_yscher hPSC_undifferentiated hPSC_Novo_Nordisk LARRY CellTag3 SpaceBar DTT_mouse)
echo "[$(date)] start ${#DS[@]} datasets"
printf '%s\n' "${DS[@]}" | xargs -P 8 -I{} sh -c '$PY pair_distance.py --no-combine {} > $LOG/pairdist_{}.out 2> $LOG/pairdist_{}.err; rc=$?; echo "{} exit $rc"; exit $rc'
rc=$?
$PY pair_distance.py combine
echo "[$(date)] done (xargs exit $rc; nonzero = at least one dataset failed, see logs/pairdist_<dataset>.err)"
exit $rc
