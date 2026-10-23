#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --time=2:00:00
#SBATCH --job-name=fm01_flow
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm01/data/top100_C_U_scenarios/logs/%x_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm01/data/top100_C_U_scenarios/logs/%x_%j.err
# FM01 flowchart scenarios for the top-100 pairs by |C| and by |U| (~190 unique pairs). 24 independent chunks (pair i mod 24), 2000 null draws per step,
# ~2 min/pair single-core (tested: 62 s/pair at 1000 draws) -> ~8 pairs x 2 min per chunk. Env python called directly; outputs on shared storage.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
R=/gpfs/projects/b1255/hzhang/TwINFER_KA
source $R/code/TwINFER/env.sh
O=$R/analysis_data/fm01/data/top100_C_U_scenarios
cd $R/code/TwINFER/real_data_scripts/FM01_twinscore
for i in $(seq 0 23); do
  /home/gzu5140/.conda/envs/twinfer-code/bin/python 23_flowchart_scenarios_top100_C_U.py $O/flowchart_chunk_$i.csv --n-shuffles 2000 --chunk $i/24 > $O/logs/chunk_$i.log 2>&1 &
done
wait
