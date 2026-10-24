#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=rebuild_fullsheet
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
# Figure 5 preprocessing: FM01 and FM06 rebuilt from the raw GEO matrices and the barcode tables with the paper mito cutoffs (FM01 26 %, FM06 21 %),
# singletCode run ONCE on the whole barcode sheet and the criterion-4 rescue per (sample, cellID). Output: <REBUILD>/FateMap/<FM>_data/qc_filtered
# (final_data is never touched; compare with final_data/<FM>_data/qc_filtered afterwards). Logs: <REBUILD>/logs.
# Usage: sbatch run_rebuild_fullsheet.sbatch      (paths: see fig5_paths.py; override with TWINFER_PROJECT_ROOT, FIG5_RAW_DATA, FIG5_REBUILD_DIR)
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
cd "$CODE/FateMap_FM01_FM06_Watermelon"
for ds in FM01 FM06; do $PY build_fm_fullsheet_rebuild.py $ds "$REBUILD/FateMap" > "$LOGS/rebuild_fullsheet_${ds}_${SLURM_JOB_ID}.log" 2>&1 & done
wait
for ds in FM01 FM06; do echo $ds; tail -22 "$LOGS/rebuild_fullsheet_${ds}_${SLURM_JOB_ID}.log" | grep -E "raw=|n_rescued|n_clones"; done
