#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --job-name=het_vs_reg
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k/logs/slurmLog-%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/heterogeneity_vs_regulation_1k/logs/slurmLog-%j.err

set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# Avoid BLAS thread oversubscription fighting with joblib's process-level parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# cd /gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/heterogeneity_vs_regulation   # [2026-09-30 replaced by env.sh variable]
# cd ${TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/heterogeneity_vs_regulation   # [2026-09-30 replaced by env.sh variable]
cd "${TWINFER_CODE_ROOT}/paper_analysis/heterogeneity_vs_regulation"

echo "Job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK cpus, start $(date)"

# ~/.conda/envs/twinfer-code/bin/python -u run_analysis.py --n_jobs "$SLURM_CPUS_PER_TASK"   # [2026-09-30 replaced by env.sh variable]
# "${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" -u run_analysis.py --n_jobs "$SLURM_CPUS_PER_TASK"   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-~/.conda/envs/twinfer-code/bin/python}" -u run_analysis.py --n_jobs "$SLURM_CPUS_PER_TASK"

echo "Job $SLURM_JOB_ID end $(date)"
