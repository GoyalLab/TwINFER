#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=mich_rerun
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Figure 5 verification of the Michaels T1 / T2 clone labels: independent rerun of yscher's lineage-barcode pipeline (steps 1-7 of michaels_barcodes.py) from the TLB FASTQs,
# compared with yscher's cell_to_clone.csv. Output: <REBUILD>/Michaels_T1T2/michaels_rerun/{steps.json, validation.json, final_cell_barcode_umis.csv.gz, ...}.
# Step 8 (graph -> clones) is checked separately: michaels_step8_seeds.py (20 hash seeds).  Usage: sbatch run_michaels_rerun.sbatch
set -uo pipefail
ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
FIG=$ROOT/final_figure_data/real_data_analysis_and_melanoma
RAW=${FIG5_RAW_DATA:-$FIG/raw_data}                                  # links to the raw data (see raw_data/)
FINAL=$FIG/final_data                                                # links to the final objects
REBUILD=${FIG5_REBUILD_DIR:-$FIG/analysis_data/preprocessing_rebuild}  # rebuilt files go here, never into final_data
CODE=$ROOT/code/TwINFER/paper_analysis/real_data_analysis_and_melanoma/preprocessing_code
PY=${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}   # conda activate does not work in batch jobs: call the environment python directly
LOGS=$REBUILD/logs; mkdir -p "$LOGS"
export TWINFER_PROJECT_ROOT=$ROOT
cd "$CODE/Michaels_T1T2"
$PY michaels_rerun_barcodes.py
