#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=fs_all
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
# Figure 5 preprocessing, ONE job: Watermelon (QC with the corrected criterion-4 rescue, split by stage, stage-level integrated h5ads) and the integrated h5ads
# of FM01 and FM06 (run_rebuild_fullsheet.sbatch must have produced their qc_filtered folders). Everything is written to <REBUILD>/FateMap/; final_data is untouched.
# (The PCA and the pair distances are the analysis step: analysis_code/transcriptomic_distance_analysis/.)  Usage: sbatch run_fs_all.sbatch
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
J=$SLURM_JOB_ID
run(){ n=$1; shift; "$@" > "$LOGS/fs_all_${n}_$J.log" 2>&1 && echo "OK $n" || echo "FAILED $n"; }
# phase 1 (parallel): Watermelon QC + split by stage | integrated FM01, FM06
( run wm_qc $PY rebuild_corrected_downstream.py wm_qc && run wm_split $PY rebuild_corrected_downstream.py wm_split ) &
for d in FM01 FM06; do run integ_$d $PY rebuild_corrected_downstream.py fm_integrated $d & done
wait
# phase 2 (needs the Watermelon split): stage-level integrated h5ads
for s in naive lag late; do run wmstage_$s $PY rebuild_corrected_downstream.py wm_stage $s & done
wait
echo "[$(date)] done"
