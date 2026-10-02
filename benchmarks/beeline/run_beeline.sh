#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
# Generic Beeline benchmark launcher (replaces ~14 near-identical run_*_benchmark / run_*_genie3_grnboost2 scripts, see _archive/benchmarks/beeline/).
# Usage:  source clean_code/env.sh ; bash run_beeline.sh <run-name> [--dry-run] [concurrency]       (run-name = file in runs/<run-name>.env)
#         sbatch <resources from runs/<run-name>.env: SBATCH_HINT> --output=<logdir>/beeline_%j.out run_beeline.sh <run-name>
# #SBATCH output/resources cannot be variables, so pass them on the sbatch command line (SBATCH_HINT in each runs/*.env records what the original script used).
# --dry-run prints the commands without running them. The Beeline algorithms need the full install ($TWINFER_BEELINE_PATH) and the BEELINE env ($BEELINE_PYTHON).
set -eo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BEELINE_PATH:?source clean_code/env.sh first (full Beeline install: BLRunner.py + Algorithms/)}"
RUN="${1:?usage: run_beeline.sh <run-name> [--dry-run] [concurrency]}"; shift
DRY=0; CONCURRENCY=15
for a in "$@"; do case "$a" in --dry-run) DRY=1;; ''|*[!0-9]*) echo "unknown argument $a" >&2; exit 2;; *) CONCURRENCY="$a";; esac; done
ENVFILE="${TWINFER_CODE_ROOT}/benchmarks/beeline/runs/${RUN}.env"
[ -f "$ENVFILE" ] || { echo "no such run: $ENVFILE" >&2; exit 2; }
# defaults, overridden by the run file
CONFIG_ABS=0; MODE=single; REVERSE=0; SKIP_POPULATED=1; REDIRECT_NAME=""; MODULES=""; BASE_CONFIG=""; SUBCONFIG_DIR_NAME=""
source "$ENVFILE"
BEELINE_PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python}"
SCRIPT_DIR="${TWINFER_BEELINE_PATH}"
LOG_DIR="${TWINFER_PROJECT_ROOT}/${LOG_DIR_REL}"
say() { echo "+ $*" >&2; }
mkdir -p "$LOG_DIR"
[ "$MODE" = per_dataset ] && mkdir -p "${SCRIPT_DIR}/config-files/${SUBCONFIG_DIR_NAME}"
if [ -n "$MODULES" ]; then say "module load $MODULES"; [ "$DRY" = 1 ] || module load $MODULES; fi
cd "$SCRIPT_DIR"
echo "[$(date)] run: $RUN ($DESC)"
if [ "$MODE" = single ]; then
    [ "$CONFIG_ABS" = 1 ] && CONFIG="${SCRIPT_DIR}/${CONFIG}"
    cmd=("$BEELINE_PYTHON" BLRunner.py --config "$CONFIG")
    [ "$REVERSE" = 1 ] && cmd+=(--reverse)
    cmd+=(--yes)
    [ "$SKIP_POPULATED" = 1 ] && cmd+=(--skip-populated)
    if [ -n "$REDIRECT_NAME" ]; then
        say "${cmd[*]} > ${LOG_DIR}/${REDIRECT_NAME}.out 2> ${LOG_DIR}/${REDIRECT_NAME}.err"
        [ "$DRY" = 1 ] || "${cmd[@]}" > "${LOG_DIR}/${REDIRECT_NAME}.out" 2> "${LOG_DIR}/${REDIRECT_NAME}.err"
    else
        say "${cmd[*]}"
        [ "$DRY" = 1 ] || "${cmd[@]}"
    fi
else
    # per_dataset: one sub-config per dataset (should_run only that dataset), BLRunner per sub-config, $CONCURRENCY in parallel
    BASE="${SCRIPT_DIR}/${BASE_CONFIG}"; SUBCONFIG_DIR="${SCRIPT_DIR}/config-files/${SUBCONFIG_DIR_NAME}"
    DATASET_IDS=$("$BEELINE_PYTHON" - "$BASE" "$SUBCONFIG_DIR" <<'PYEOF'
import sys, yaml
base_config_path, subconfig_dir = sys.argv[1], sys.argv[2]
with open(base_config_path) as f:
    config = yaml.safe_load(f)
dataset_ids = [d["dataset_id"] for d in config["input_settings"]["datasets"]]
for target_id in dataset_ids:
    variant = yaml.safe_load(yaml.dump(config))
    for d in variant["input_settings"]["datasets"]:
        d["should_run"] = [d["dataset_id"] == target_id]
    with open(f"{subconfig_dir}/{target_id}.yaml", "w") as f:
        yaml.dump(variant, f, sort_keys=False)
    print(target_id)
PYEOF
)
    run_one() {
        local dataset_id="$1"
        "$BEELINE_PYTHON" BLRunner.py \
            --config "config-files/${SUBCONFIG_DIR_NAME}/${dataset_id}.yaml" \
            --yes --skip-populated \
            > "${LOG_DIR}/full_${dataset_id}.out" 2> "${LOG_DIR}/full_${dataset_id}.err"
    }
    export -f run_one
    export BEELINE_PYTHON LOG_DIR SUBCONFIG_DIR_NAME
    if [ "$DRY" = 1 ]; then printf '%s\n' $DATASET_IDS | while read -r d; do say "$BEELINE_PYTHON BLRunner.py --config config-files/${SUBCONFIG_DIR_NAME}/${d}.yaml --yes --skip-populated > ${LOG_DIR}/full_${d}.out 2> ${LOG_DIR}/full_${d}.err"; done
    else printf '%s\n' $DATASET_IDS | xargs -P "$CONCURRENCY" -I DS bash -c 'run_one "$@"' _ DS; fi
fi
echo "[$(date)] done: $RUN"
