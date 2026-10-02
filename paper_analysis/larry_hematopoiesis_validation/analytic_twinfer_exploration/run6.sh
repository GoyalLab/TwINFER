#!/bin/bash
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
for P in detlow detmid dethigh hvglow hvgmid hvghigh; do
  echo "############# $P #############"
# PANEL=$P /home/gzu5140/.conda/envs/twinfer-code/bin/python3 analytic_variants.py 2>&1 | grep -vE "UserWarning|pkg_resources|warn"   # [2026-09-30 replaced by env.sh variable]
# PANEL=$P "${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}" analytic_variants.py 2>&1 | grep -vE "UserWarning|pkg_resources|warn"   # [2026-09-30 replaced by env.sh variable]
  PANEL=$P "${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}" analytic_variants.py 2>&1 | grep -vE "UserWarning|pkg_resources|warn"
done
