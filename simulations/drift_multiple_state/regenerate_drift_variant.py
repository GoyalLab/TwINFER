"""
Generate the drift 2-state simulations for two NEW drift regimes, both networks:

  variant "fast5h"  : same endpoints as the original drift (up 1.0->1.66x,
                      down 1.0->0.12x) but the linear k_on ramp completes in
                      tau = 5 h instead of 15 h.
  variant "recover" : k_on RISES from a suppressed level toward baseline.
                      "up"  sub-population : 0.12x -> 1.0x  over tau = 15 h  (recovering)
                      "down" sub-population: stays at 0.12x                 (no drift)

networks:
  A_B     -> connectivity_matrix_A_B.txt      (no regulation)   type A_B_no_reg_2_states_<variant>
  A_to_B  -> connectivity_matrix_A_to_B.txt   (gene_1 -> gene_2) type A_to_B_2_states_<variant>

20 replicates per (network x variant) -> 4 x 20 = 80 sims.

Output: <OUT>/df_rows_0_0_<i>_<timestamp>_ncells_6000_<type>_<id>.csv
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import glob
import os
import time

from joblib import Parallel, delayed
from numba import set_num_threads

from twinfer.simulation.gillespie_drift import process_param_set
from twinfer.utils.paths import get_repo_root

REPO = get_repo_root()
DRIFT_T_START = 1500   # simulation_time_before_division must equal the drift t_start

NETWORKS = {
    "A_B":    ("connectivity_matrix_A_B.txt",    "A_B_no_reg_2_states"),
    "A_to_B": ("connectivity_matrix_A_to_B.txt", "A_to_B_2_states"),
}

# drift-ramp overrides passed straight into process_param_set's base_config
VARIANTS = {
    "fast5h": dict(drift_tau=5,
                   drift_start_up=1.0,  drift_final_up=1.66,
                   drift_start_down=1.0, drift_final_down=0.12),
    "recover": dict(drift_tau=15,
                    drift_start_up=0.12,  drift_final_up=1.0,     # recovering -> baseline
                    drift_start_down=0.12, drift_final_down=0.12), # stays suppressed
}


def build_config(out_dir, network, variant, twin_time=48):
    conn, base_type = NETWORKS[network]
    cfg = {
        "n_cells": 6000,
        "simulation_time_before_division": DRIFT_T_START,
        "twin_simulation_time_after_division": twin_time,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": f"{REPO}/simulation_example_input_data/{conn}",
        "param_csv": f"{REPO}/simulation_example_input_data/median_parameter.csv",
        "rows_to_use": [[0, 0]],
        "output_folder": out_dir,
        "log_file": f"{out_dir}/log.jsonl",
        "type": f"{base_type}_{variant}",
        "drift_t_start": DRIFT_T_START,
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": 4,
    }
    cfg.update(VARIANTS[variant])
    return cfg


def _exists(out_dir, i, type_name):
    return bool(glob.glob(os.path.join(
        out_dir, f"df_rows_0_0_{i}_*_ncells_*_{type_name}_*.csv")))


def run_rep(i, cfg, skip_existing=True):
    if skip_existing and _exists(cfg["output_folder"], i, cfg["type"]):
        print(f"[{cfg['type']} rep {i}] skip (exists)", flush=True)
        return None
    set_num_threads(cfg["number_of_cores_per_parameter"])
    t0 = time.time()
    path = process_param_set(cfg["rows_to_use"][0], f"rows_0_0_{i}", cfg)
    print(f"[{cfg['type']} rep {i}] {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, choices=list(NETWORKS))
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--out-dir",
                    default=f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation_variants')
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--cores-per", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=6000, help="smoke-test override")
    ap.add_argument("--twin-time", type=int, default=48,
                    help="hours of twin simulation after division (hourly measurements)")
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = build_config(args.out_dir, args.network, args.variant, twin_time=args.twin_time)
    cfg["number_of_cores_per_parameter"] = args.cores_per
    cfg["n_cells"] = args.n_cells
    skip = not args.no_skip_existing

    print(f"Generating {args.reps} reps  network={args.network}  variant={args.variant}  "
          f"type={cfg['type']}  -> {args.out_dir}  "
          f"({args.jobs} parallel x {args.cores_per} cores)", flush=True)
    print(f"  drift overrides: {VARIANTS[args.variant]}", flush=True)

    Parallel(n_jobs=args.jobs, backend="multiprocessing")(
        delayed(run_rep)(i, cfg, skip) for i in range(1, args.reps + 1)
    )
    print("done", flush=True)
