"""
Regenerate the A_B (no-regulation) 2-state drift simulation, 20 replicates.

Mirrors simulate_drift_multiple_states.py's base_config_list[1] (the A_B case)
but calls process_param_set from the packaged module
twinfer.simulation.gillespie_drift directly, once per replicate i = 1..20.

Output: <OUT>/df_rows_0_0_<i>_<timestamp>_ncells_6000_A_B_no_reg_2_states_<id>.csv
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import glob
import os
import time
import numpy as np
from joblib import Parallel, delayed
from numba import set_num_threads

from twinfer.simulation.gillespie_drift import process_param_set
from twinfer.utils.paths import get_repo_root

REPO = get_repo_root()  # <repo>/code/TwINFER

# gillespie_drift.process_param_set hardcodes the drift ramp to start at global
# time t_start=1500 (lines ~837-849). The twins use t_offset = t_parent_end =
# simulation_time_before_division. For the drift to begin EXACTLY at division
# (twin time 0), simulation_time_before_division must equal this t_start.
DRIFT_T_START = 1500


def build_config(out_dir):
    return {
        "n_cells": 6000,
        "simulation_time_before_division": DRIFT_T_START,   # must == drift t_start
        "twin_simulation_time_after_division": 48,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": f"{REPO}/simulation_example_input_data/connectivity_matrix_A_B.txt",
        "param_csv": f"{REPO}/simulation_example_input_data/median_parameter.csv",
        "rows_to_use": [[0, 0]],
        "output_folder": out_dir,
        "log_file": f"{out_dir}/log.jsonl",
        "type": "A_B_no_reg_2_states",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": 4,
    }


def _exists(out_dir, i):
    return bool(glob.glob(os.path.join(
        out_dir, f"df_rows_0_0_{i}_*_ncells_*_A_B_no_reg_2_states_*.csv")))


def run_rep(i, cfg, skip_existing=True):
    if skip_existing and _exists(cfg["output_folder"], i):
        print(f"[rep {i}] skip (already exists)", flush=True)
        return None
    set_num_threads(cfg["number_of_cores_per_parameter"])
    label = f"rows_0_0_{i}"
    t0 = time.time()
    path = process_param_set(cfg["rows_to_use"][0], label, cfg)
    print(f"[rep {i}] {(time.time()-t0)/60:.1f} min ({cfg['number_of_cores_per_parameter']} cores) -> {path}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation')
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rep", type=int, default=None,
                    help="run ONLY this single replicate index (overrides --reps/--jobs)")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--cores-per", type=int, default=4)
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="regenerate even if df_rows_0_0_<i>_*A_B_no_reg* already exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = build_config(args.out_dir)
    cfg["number_of_cores_per_parameter"] = args.cores_per
    skip = not args.no_skip_existing

    if args.rep is not None:
        print(f"Single replicate {args.rep} -> {args.out_dir} "
              f"({args.cores_per} cores, skip_existing={skip})", flush=True)
        run_rep(args.rep, cfg, skip)
        print("done", flush=True)
        raise SystemExit(0)

    print(f"Generating {args.reps} A_B (no-reg) drift replicates -> {args.out_dir} "
          f"({args.jobs} parallel x {args.cores_per} cores, skip_existing={skip})", flush=True)

    Parallel(n_jobs=args.jobs, backend="multiprocessing")(
        delayed(run_rep)(i, cfg, skip) for i in range(1, args.reps + 1)
    )
    print("done", flush=True)
