#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 8:00:00
#SBATCH --job-name=sb_cab_comp
#SBATCH --array=0-2
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/sb_cab_comp_%A_%a.out
#SBATCH --output=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/sb_cab_comp_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/sb_cab_comp_%A_%a.err
#SBATCH --error=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/sb_cab_comp_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

GENE_SETS=(correlation_high correlation_mid correlation_low)
GS=${GENE_SETS[$SLURM_ARRAY_TASK_ID]}

# write the same (deterministic) centroid A/B input the TwINFER job uses, so competitors see identical cells
"$PYTHON" - <<PY
from run_infer_spacebar_centroid_ab import build_twinfer_input_centroid_ab
df = build_twinfer_input_centroid_ab("$GS")
# df.to_csv("/home/gzu5140/TwINFER_KA/analysis_data/spacebar/data/twinfer_input_spacebar_${GS}_centroid_ab.csv", index=False)   # [2026-09-30 replaced by env.sh variable]
df.to_csv("${TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data/twinfer_input_spacebar_${GS}_centroid_ab.csv", index=False)
PY

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: SpaceBar centroid A/B competitors / $GS"
"$PYTHON" run_competitors_fatemap.py SPACEBAR --gene-set "${GS}_centroid_ab" --n-cores 8
echo "[$(date)] Done"
