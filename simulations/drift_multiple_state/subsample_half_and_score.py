"""
Subsampling-noise study: for each of the 8 scenarios' existing 3 replicates,
draw K independent random halves of the clone population (no resimulation --
just filter the already-simulated CSV) and rescore each half with the exact
same Step 1-4 + gated-regulation pipeline as run_6scenario_zscores.py.

Purpose: separate pure population-size noise (spread across the K half-draws
of ONE rep) from rep-to-rep simulation noise (spread across the 3 full reps,
already in eight_scenario/t*/six_scenario_zscores.csv).

Scenario names get a "_half" suffix so they sit next to the full-population
rows when concatenated for plotting.

Usage:
  python subsample_half_and_score.py --output-dir DIR --draws 5 --n-cores 2 --jobs 3
"""
import zlib  # [2026-10-01 added]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
import argparse
import glob
import json
import os
import re
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from simulations.drift_multiple_state import run_6scenario_zscores as R

# TMP_DIR = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/058d7e5b-b617-41cc-9f79-a9cc24c4cfb8/scratchpad/half_subsample_csvs"   # [2026-09-30 replaced: pointed at an ephemeral Claude session scratchpad; now a stable dir on shared storage]
TMP_DIR = f"{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/half_subsample_csvs"

FIRST3 = {
    "no_regulation": [0, 1, 2], "A_to_B": [0, 1, 2],
    "multistate_A_B": [1, 2, 3], "multistate_A_to_B": [1, 2, 3],
    "kramp_A_B": [1, 2, 3], "kramp_A_to_B": [1, 2, 3],
    "kfrozen_A_B": [1, 2, 3], "kfrozen_A_to_B": [1, 2, 3],
}


def _half_csv(path, scen, rep, draw, seed):
    df = R._load(path)
    clones = np.sort(df["clone_id"].unique())
    rng = np.random.default_rng(seed)
    half = rng.choice(clones, size=len(clones) // 2, replace=False)
    df_sub = df[df["clone_id"].isin(half)].reset_index(drop=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    out = os.path.join(TMP_DIR, f"{scen}_rep{rep}_half{draw}.csv")
    df_sub.to_csv(out, index=False)
    return out


def run_one(scen, rep, draw, path, out_dir, n_cores):
    tag = f"{scen}_half"
    out = os.path.join(out_dir, f"{tag}_rep_{rep}_d{draw}.json")
    if os.path.exists(out):
        return out, "skipped"
    try:
        # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] seed = hash((scen, rep, draw)) & 0xFFFFFFFF
        seed = zlib.crc32(repr((scen, rep, draw)).encode()) & 0xFFFFFFFF
        half_path = _half_csv(path, scen, rep, draw, seed)
        rec = R.process_replicate(half_path, tag, f"{rep}_d{draw}", n_cores=n_cores)
        os.makedirs(out_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, cls=R.NpEnc, indent=2)
        os.remove(half_path)
        def _f(x):
            return "None" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"
        return out, f"OK  zreg_gated(t1/t2)={_f(rec['z_reg_gated_t1'])}/{_f(rec['z_reg_gated_t2'])}"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return out, f"ERROR {type(e).__name__}: {e}"


def _aggregate(out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*_half_rep_*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["scenario", "rep_id"])
    csv = os.path.join(out_dir, "half_subsample_zscores.csv")
    df.to_csv(csv, index=False)
    print(f"\naggregated {len(df)} rows -> {csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--n-cores", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()

    all_tasks = R.collect_tasks()
    by_scen_rep = {}
    for scen, rep, path in all_tasks:
        if scen in FIRST3 and rep in FIRST3[scen]:
            by_scen_rep[(scen, rep)] = path

    tasks = []
    for (scen, rep), path in sorted(by_scen_rep.items()):
        for draw in range(args.draws):
            tasks.append((scen, rep, draw, path))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(tasks)} half-subsample tasks ({len(by_scen_rep)} scenario-reps x {args.draws} draws) "
          f"-> {args.output_dir}", flush=True)

    def _do(i, scen, rep, draw, path):
        out, status = run_one(scen, rep, draw, path, args.output_dir, args.n_cores)
        print(f"[{i+1}/{len(tasks)}] {scen} rep{rep} draw{draw}: {status}", flush=True)

    Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(_do)(i, s, r, d, p) for i, (s, r, d, p) in enumerate(tasks))

    _aggregate(args.output_dir)
    print("done", flush=True)
