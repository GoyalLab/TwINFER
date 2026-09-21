#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 8GB
#SBATCH -t 1:00:00
#SBATCH --array=0-11
#SBATCH --job-name=ns_final_allpairs_t1_10
#SBATCH --output=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/ns_final_allpairs_t1_10_%A_%a.out
#SBATCH --error=/home/gzu5140/TwINFER_KA/analysis_data/larry_barcode_extraction/logs/ns_final_allpairs_t1_10_%A_%a.err
set -euo pipefail

# t1=10,t2=20 rerun of infer_network_sweep_final_allpairs.py (measure metrics farther from the
# division event). 148 replicates, tested at ~54s/replicate locally -> chunk of 13 x 54s ~= 12min,
# 1h budget for headroom (same array shape as the original run_network_sweep_final_allpairs.sh).
# Output: analysis_data/network_sweep_final/twinfer_inference_allpairs_t1_10/

PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3
cd /home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark

CHUNK=13
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
echo "[$(date)] array task $SLURM_ARRAY_TASK_ID: tasks[$START:$((START+CHUNK))]"
"$PYTHON" infer_network_sweep_final_allpairs_t1_10.py --n-cores 4 --start "$START" --count "$CHUNK"
echo "[$(date)] Done"
