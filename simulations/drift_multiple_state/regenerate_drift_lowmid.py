"""
Rebuild of the low->mid 2-state drift, both "kinds", both networks.

Absolute k_on levels:  low = 0.12   mid = 0.66   high = 1.66
Burn-in runs at k_on = low; K is calibrated there.  At division the population
splits 50/50: one half stays at k_on = low, the other reaches k_on = mid.

  kind K_frozen : mid half's k_on RAMPS low->mid over tau; K never changes.
  kind K_ramp   : mid half's k_on ramps low->mid AND its g1->g2 Hill K ramps
                  Hill K is recomputed for k_on = mid (only that half changes).

Output: <OUT>/df_rows_0_0_<i>_<ts>_ncells_6000_<net>_lowmid_<kind>_<id>.csv
        (state column = "low" / "mid")
"""
import argparse
import glob
import os
import time

from joblib import Parallel, delayed
from numba import set_num_threads

from twinfer.simulation.gillespie_drift import process_param_set_lowmid
from twinfer.utils.paths import get_repo_root

REPO = get_repo_root()
DRIFT_T_START = 1500

K_ON = {"low": 0.12, "mid": 0.66, "high": 1.66}

NETWORKS = {
    "A_B":    ("connectivity_matrix_A_B.txt",    "A_B_no_reg"),
    "A_to_B": ("connectivity_matrix_A_to_B.txt", "A_to_B"),
}


def build_config(out_dir, network, kind, twin_time, kfrozen_dir=None):
    conn, base_type = NETWORKS[network]
    cfg = {
        "n_cells": 6000,
        "simulation_time_before_division": DRIFT_T_START,
        "twin_simulation_time_after_division": twin_time,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": f"{REPO}/simulation_example_input_data/{conn}",
        "param_csv": f"{REPO}/simulation_example_input_data/median_parameter.csv",
        "rows_to_use": [[0, 0]],          # k_on gets overridden to drift_k_on_burnin anyway
        "output_folder": out_dir,
        "log_file": f"{out_dir}/log.jsonl",
        "type": f"{base_type}_lowmid_{kind}",
        "drift_kind": kind,               # "K_frozen" | "K_ramp"
        "drift_k_on_burnin": K_ON["low"],
        "drift_k_on_low_final": K_ON["low"],
        "drift_k_on_mid_final": K_ON["mid"],
        "drift_t_start": DRIFT_T_START,
        "drift_tau": 15,
        "number_of_cores_per_parameter": 4,
    }
    # optional: reuse the k_on=0.12 burn-in end state from a K_frozen run
    cfg["_burnin_dir"] = kfrozen_dir
    cfg["_burnin_type"] = f"{base_type}_lowmid_K_frozen"
    return cfg


def _exists(out_dir, i, type_name):
    return bool(glob.glob(os.path.join(out_dir, f"df_rows_0_0_{i}_*_{type_name}_*.csv")))


def run_rep(i, cfg, skip_existing=True):
    if skip_existing and _exists(cfg["output_folder"], i, cfg["type"]):
        print(f"[{cfg['type']} rep {i}] skip (exists)", flush=True)
        return None
    set_num_threads(cfg["number_of_cores_per_parameter"])
    if cfg.get("_burnin_dir"):
        pat = os.path.join(cfg["_burnin_dir"],
                           f"simulation_before_division_df_rows_0_0_{i}_*_{cfg['_burnin_type']}_*.csv")
        hits = glob.glob(pat)
        if hits:
            cfg = {**cfg, "burnin_csv": sorted(hits)[0]}
        else:
            print(f"[{cfg['type']} rep {i}] no reusable burn-in ({pat}) -> running full burn-in", flush=True)
    t0 = time.time()
    path = process_param_set_lowmid(cfg["rows_to_use"][0], f"rows_0_0_{i}", cfg)
    print(f"[{cfg['type']} rep {i}] {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, choices=list(NETWORKS))
    ap.add_argument("--kind", required=True, choices=["K_frozen", "K_ramp"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--cores-per", type=int, default=4)
    ap.add_argument("--twin-time", type=int, default=300)
    ap.add_argument("--n-cells", type=int, default=6000, help="smoke-test override")
    ap.add_argument("--kfrozen-dir", default=None,
                    help="reuse the k_on=0.12 burn-in end states from this K_frozen output dir")
    ap.add_argument("--k-meanfield-dt", type=float, default=0.05,
                    help="K_ramp: hours between live mean-field K updates")
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = build_config(args.out_dir, args.network, args.kind, args.twin_time, kfrozen_dir=args.kfrozen_dir)
    cfg["number_of_cores_per_parameter"] = args.cores_per
    cfg["n_cells"] = args.n_cells
    cfg["K_meanfield_dt"] = args.k_meanfield_dt
    skip = not args.no_skip_existing

    print(f"Generating {args.reps} reps  net={args.network}  kind={args.kind}  "
          f"type={cfg['type']}  twin-time={args.twin_time}  -> {args.out_dir}", flush=True)

    Parallel(n_jobs=args.jobs, backend="multiprocessing")(
        delayed(run_rep)(i, cfg, skip) for i in range(1, args.reps + 1))
    print("done", flush=True)
