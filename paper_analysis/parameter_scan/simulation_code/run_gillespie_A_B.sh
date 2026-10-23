#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --mem=10GB
#SBATCH --time=48:00:00
#SBATCH --job-name=scan_A_B
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/logs/parameter_scan_A_B_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/logs/parameter_scan_A_B_%A_%a.err
#SBATCH --array=0-41

# Parameter-scan simulations, topology A_B (no regulation, matrix 0,0;0,0).
#
# Figure     : parameter_scan (used by heterogeneity_vs_regulation and causal_direction_inference).
# What it does: simulates 25,000 parameter sets (Latin hypercube, parameter_generation_for_simulation.ipynb) of the 2-gene system A_B:
#                6,000 cells, 1500 h burn-in, twins for 48 h after division, sampled hourly. One output file per parameter set.
#                Parameters come from parameters_3genes_positive_reg_pi_on_r_add_scaled.csv; the network from connectivity_matrix_A_B.txt (both in input_folder).
# Array tasks : task k simulates parameter sets 600k ... 600k+599 (set 24,999 is the last), so --array=0-41 covers all 25,000 sets.
#                Sets whose output file already exists are skipped, so a stopped array can simply be resubmitted.
# Environment : OUT_ROOT           output root (default final_figure_data/parameter_scan/simulation_data/parameter_scan_raw);
#                                  files go to $OUT_ROOT/A_B/ and logs to $OUT_ROOT/logs/
#               SAVE_BEFORE_DIVISION=1  also write the large simulation_before_division_* file (default: not written)
#               SET_LIST_FILE      a text file with one parameter-set id per line; then array task k simulates only the id on line k+1
#                                  (use it to re-simulate chosen sets, e.g.  SET_LIST_FILE=ids.txt sbatch --array=0-13 run_gillespie_A_B.sh)
# Usage       : source code/TwINFER/env.sh          # sets TWINFER_PROJECT_ROOT, TWINFER_PYTHON
#               sbatch run_gillespie_A_B.sh
set -euo pipefail
: "${TWINFER_PROJECT_ROOT:?source code/TwINFER/env.sh before submitting}"
PY="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}"

DATA="$TWINFER_PROJECT_ROOT/final_figure_data/parameter_scan"
INPUT="$DATA/input_folder"
OUT_ROOT="${OUT_ROOT:-$DATA/simulation_data/parameter_scan_raw}"
OUT="$OUT_ROOT/A_B"
mkdir -p "$OUT" "$OUT_ROOT/logs"

N_SETS=25000
CHUNK=600
if [ -n "${SET_LIST_FILE:-}" ]; then
    first=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SET_LIST_FILE")
    last=$first
    [ -n "$first" ] || { echo "no id on line $((SLURM_ARRAY_TASK_ID + 1)) of $SET_LIST_FILE"; exit 0; }
else
    first=$((CHUNK * SLURM_ARRAY_TASK_ID))
    last=$((first + CHUNK - 1))
    [ "$last" -lt "$N_SETS" ] || last=$((N_SETS - 1))
fi
# row_to_start / row_to_end are parameter-set ids and both are INCLUSIVE
extra=""
[ "${SAVE_BEFORE_DIVISION:-0}" = "1" ] || extra="--no_save_before_division"

echo "A_B: parameter sets $first-$last -> $OUT  ($(date))"
"$PY" -m twinfer.simulation.gillespie_simulations \
    --path_to_connectivity_matrix "$INPUT/connectivity_matrix_A_B.txt" \
    --param_csv "$INPUT/parameters_3genes_positive_reg_pi_on_r_add_scaled.csv" \
    --row_to_start "$first" \
    --row_to_end "$last" \
    --output_folder "$OUT" \
    --log_file "$OUT_ROOT/logs/A_B.jsonl" \
    --type "A_B" \
    --number_parallel_processes 2 \
    --number_of_cores_per_parameter 16 \
    --n_genes 2 \
    --n_cells 6000 \
    --simulation_time_before_division 1500 \
    --twin_simulation_time_after_division 48 \
    --twin_measurement_resolution 1 \
    $extra
echo "A_B: done  ($(date))"
