#!/bin/bash
#SBATCH --job-name=michaels_qc_repro
#SBATCH --account=b1042
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/real_data/Michaels_et_al/qc_repro_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/real_data/Michaels_et_al/qc_repro_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=genomics
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"

# /home/gzu5140/.conda/envs/twinfer-code/bin/python -u \   # [2026-09-30 replaced by env.sh variable]
# "${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}" -u \   # [2026-09-30 replaced by env.sh variable]
# /gpfs/projects/b1255/hzhang/TwINFER_KA/real_data/Michaels_et_al/reproduce_timecourse_qc.py   # [2026-09-30 replaced by env.sh variable]
"${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python}" -u \
  ${TWINFER_PROJECT_ROOT}/real_data/Michaels_et_al/reproduce_timecourse_qc.py
