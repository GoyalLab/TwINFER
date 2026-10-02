#!/bin/bash
# SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version).
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 32GB
#SBATCH -t 4:00:00
#SBATCH --job-name=supp_boot_fm06
#SBATCH --array=0-8
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/supp_boot_fm06_%A_%a.out
#SBATCH --output=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/supp_boot_fm06_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline/logs/supp_boot_fm06_%A_%a.err
#SBATCH --error=/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/logs/supp_boot_fm06_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/fatemap_pipeline"

# 2026-09-23: rerun FM06's A/B-split TwinScore_supplement with genuine bootstrap noise floors
# (bootstrap_null_sd_diag/offdiag) replacing the analytic null_sd(meff) everywhere it fed a
# significance/floor threshold (phi's h-floor, sd_reg, sd_twin_t2) -- see conversation for why
# (null_sd checked directly against a real permutation null and found to underestimate by 1.5x-2.9x).
GENE_SETS=(variability_high variability_mid variability_low \
           detection_high detection_mid detection_low \
           correlation_high correlation_mid correlation_low)
GENE_SET=${GENE_SETS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: FM06 / $GENE_SET (bootstrap noise floors, A/B split, no gates)"
"$PYTHON" SUPERSEDED_apply_twinscore_supplement_fatemap_allpairs.py FM06 --gene-set "$GENE_SET" --n-shuffles 2000 --n-cores 8 --absplit
echo "[$(date)] Done"
