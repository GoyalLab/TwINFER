#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=fs_all
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/fs_all_%j.out
# ONE job (no arrays): build, then PCA + pair distance. Everything new goes to finalized_data/rebuild_fullsheet_2026-10-16/ ; finalized_data/<ds> untouched.
# Corrected criterion-4 rescue (cell = sample+cellID) for Watermelon, LARRY, FM05; integrated h5ads for FM01/06/08 and Watermelon stages.
cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/real_data_scripts/dataset_qc_verification
source ../../env.sh
PY=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
export QCV_FIN=/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/rebuild_fullsheet_2026-10-16
L=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs; J=$SLURM_JOB_ID
run(){ n=$1; shift; "$@" > $L/fs_all_${n}_$J.log 2>&1 && echo "OK $n" || echo "FAILED $n"; }
# phase 1 (parallel): Watermelon qc+split | LARRY stage2 + annotation | FM05 | integrated FM01 FM06 FM08
( run wm_qc $PY rebuild_corrected_downstream.py wm_qc && run wm_split $PY rebuild_corrected_downstream.py wm_split ) &
( run larry $PY rebuild_corrected_downstream.py larry && ( cd ../.. && run larry_annot env SOURCE=$QCV_FIN/LARRY_data/filtered KNN=15 $PY paper_analysis/larry_hematopoiesis_validation/preprocessing/larry_raw_annotate.py ) ) &
( cd ../.. && run fm05 $PY -m paper_analysis.fatemap_pipeline.build_fm_fullsheet_rebuild FM05 $QCV_FIN ) &
for d in FM01 FM06 FM08; do run integ_$d $PY rebuild_corrected_downstream.py fm_integrated $d & done
wait
# phase 2 (needs the Watermelon split): stage-level integrated h5ads
for s in naive lag late; do run wmstage_$s $PY rebuild_corrected_downstream.py wm_stage $s & done
wait
# phase 3: 5k-HVG PCA, then clonal-vs-random pair distance (all 50 PCs), on the new files only; 3 datasets at a time
export QCV_OUT=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/fullsheet_downstream
mkdir -p $QCV_OUT
DS="FM01 FM03 FM05 FM06 FM08 LARRY Watermelon_naive Watermelon_lag Watermelon_late"
echo "[$(date)] pca5k"; echo $DS | tr ' ' '\n' | xargs -P 3 -I{} sh -c "$PY pca5k.py {} > $L/fs_all_pca_{}_$J.log 2>&1 && echo OK pca {} || echo FAILED pca {}"
echo "[$(date)] pair_distance"; echo $DS | tr ' ' '\n' | xargs -P 3 -I{} sh -c "$PY pair_distance.py {} > $L/fs_all_pd_{}_$J.log 2>&1 && echo OK pd {} || echo FAILED pd {}"
echo "[$(date)] done"
