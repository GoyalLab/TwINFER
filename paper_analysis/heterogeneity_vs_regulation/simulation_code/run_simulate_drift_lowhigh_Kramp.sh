#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --job-name=drift_lowhigh
# original log paths: /projects/b1255/hzhang/TwINFER_KA/analysis_data/drift_lowhigh_Kramp/logs/%A_%a.{out,err}
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/logs/drift_lowhigh_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/logs/drift_lowhigh_%A_%a.err

# SLURM array job: drift simulations for the heterogeneity_vs_regulation figure (A_B and A_to_B, 1,000 replicates each).
#
# What it does : runs simulate_drift_lowhigh_Kramp.py (see its header for the model): k_on of the two halves of the 6,000 cell pairs drifts
#                from 0.66 to 0.12 (low half) and to 1.66 (high half) over 10 h after division, with the gene_1 -> gene_2 Hill K following
#                the live mean-field value; twins are then followed for 48 h. The burn-in states are taken from the single-state
#                simulations (simulation_data/single_and_two_state_simulations) so replicate n starts from the same parents.
# Array tasks  : 0 = A_B replicates 0-999,  1 = A_to_B replicates 0-499,  2 = A_to_B replicates 500-999.
#                Each task runs 12 replicates in parallel with 4 numba threads each. Re-submitting skips replicates that already exist.
# Output       : final_figure_data/heterogeneity_vs_regulation/simulation_data/drift_lowhigh_Kramp/<A_B|A_to_B>/ (df_rows_* files and
#                log.jsonl with the settings of every replicate). Logs: simulation_data/logs/drift_lowhigh_<array>_<task>.{out,err}
# Resources    : 48 CPUs, 64 GB, 6 h per task (a replicate took 1-2.5 minutes).
# Usage        : source code/TwINFER/env.sh          # sets TWINFER_CODE_ROOT, TWINFER_PROJECT_ROOT, TWINFER_PYTHON
#                sbatch run_simulate_drift_lowhigh_Kramp.sh
set -e
: "${TWINFER_CODE_ROOT:?source code/TwINFER/env.sh before submitting}"
: "${TWINFER_PROJECT_ROOT:?source code/TwINFER/env.sh before submitting}"
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
# cd ${TWINFER_CODE_ROOT}/simulations/drift_multiple_state   # original location
cd ${TWINFER_CODE_ROOT}/paper_analysis/heterogeneity_vs_regulation/simulation_code
case $SLURM_ARRAY_TASK_ID in
  0) NET=A_B;    REPS=0-999 ;;
  1) NET=A_to_B; REPS=0-499 ;;
  2) NET=A_to_B; REPS=500-999 ;;
esac
# OUT=${TWINFER_PROJECT_ROOT}/simulation_data/drift_lowhigh_Kramp/$NET   # original
OUT=${TWINFER_PROJECT_ROOT}/final_figure_data/heterogeneity_vs_regulation/simulation_data/drift_lowhigh_Kramp/$NET
echo "=== [$SLURM_ARRAY_TASK_ID] $NET reps $REPS  $(date) ==="
"$PY" simulate_drift_lowhigh_Kramp.py --network "$NET" --out-dir "$OUT" --reps "$REPS" \
    --jobs 12 --cores-per 4 --twin-time 48 --tau 10 --burnin-hours 1000 --dt-K 0.05
echo "=== [$SLURM_ARRAY_TASK_ID] DONE  $(date) ==="
