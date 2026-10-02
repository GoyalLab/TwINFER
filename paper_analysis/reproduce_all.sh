#!/bin/bash
# Run every figure folder listed in figures_manifest.yaml through run_figure.sh (written 2026-10-01, REORG_CHECKLIST 2.11).
# Usage:  source clean_code/env.sh
#         bash paper_analysis/reproduce_all.sh [--dry-run] [--validate] [--variant v2] [--stage simulate|analyze|plot|all]
#   shared_scripts (parameter_space_scan, grnboost2_comparison) run first because two figures consume their output, then the numbered figures.
#   --validate   after the runs, compare the produced numbers with paper_analysis/expected_values.tsv (columns: figure, metric, expected, tolerance, file, column, row_filter).
#                That file must be filled in by the author from the manuscript's AUPRC/F1/precision-recall values: it does not exist yet, so --validate currently
#                stops with an error rather than reporting a pass it cannot check.
# Nothing here submits SLURM jobs. Simulation stages can take hours: use --dry-run first.
set -euo pipefail
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh first}"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER_DIR="${TWINFER_PAPER_DIR:-${TWINFER_CODE_ROOT}/paper_analysis}"      # override only for tests
PYTHON="${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}"
EXTRA=(); VALIDATE=0; STAGE=all
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) EXTRA+=(--dry-run);; --variant) EXTRA+=(--variant "${2:?}"); shift;; --validate) VALIDATE=1;; --stage) STAGE="${2:?}"; shift;;
  *) echo "unknown argument: $1" >&2; exit 2;; esac; shift; done
FOLDERS=$("$PYTHON" - "$PAPER_DIR/figures_manifest.yaml" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
out = [s["folder"].replace("paper_analysis/", "") for s in m.get("shared_scripts", []) if s.get("folder")]
out += [f["folder"] for f in m.get("figures", []) if f.get("folder")]
seen = []
[seen.append(x) for x in out if x not in seen]
print("\n".join(seen))
PY
)
for f in $FOLDERS; do echo "=== $f" >&2; bash "${SELF_DIR}/run_figure.sh" "$f" "$STAGE" "${EXTRA[@]+"${EXTRA[@]}"}"; done
if [ "$VALIDATE" = 1 ]; then
  EXP="${PAPER_DIR}/expected_values.tsv"
  [ -f "$EXP" ] || { echo "--validate: $EXP does not exist (the manuscript's expected values have not been entered); nothing was validated." >&2; exit 3; }
  "$PYTHON" - "$EXP" <<'PY'
import sys, pandas as pd, numpy as np
bad = 0
for r in pd.read_csv(sys.argv[1], sep="\t").itertuples():
    df = pd.read_csv(r.file, sep=None, engine="python")
    if isinstance(r.row_filter, str) and r.row_filter.strip(): df = df.query(r.row_filter)
    got = float(df[r.column].mean()); ok = abs(got - float(r.expected)) <= float(r.tolerance); bad += (not ok)
    print(("OK   " if ok else "DIFF "), r.figure, r.metric, "expected", r.expected, "got", round(got, 6))
sys.exit(1 if bad else 0)
PY
fi
