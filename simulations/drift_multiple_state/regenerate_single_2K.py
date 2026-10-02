"""
Single-state A_B / A_to_B at the median k_on = 0.66, with the gene_1->gene_2
Hill K calibrated either at k_on = 0.66 (correct / native) or at k_on = 0.12
(deliberately too low -> for A_to_B the edge sits saturated).  3 reps each.

Output: <OUT>/df_rows_0_0_<i>_<ts>_ncells_6000_<net>_single_Kcalib_<k_on>_<id>.csv
"""
import argparse
import glob
import os
import time

from joblib import Parallel, delayed
from numba import set_num_threads

from twinfer.simulation.gillespie_drift import process_param_set_single
from twinfer.utils.paths import get_repo_root

REPO = get_repo_root()
NETWORKS = {
    "A_B":    ("connectivity_matrix_A_B.txt",    "A_B_no_reg"),
    "A_to_B": ("connectivity_matrix_A_to_B.txt", "A_to_B"),
}
K_ON_MEDIAN = 0.66
K_ON_LOW = 0.12


def build_config(out_dir, network, k_calib, twin_time):
    conn, base_type = NETWORKS[network]
    tag = f"{base_type}_single_Kcalib_{str(k_calib).replace('.', 'p')}"
    return {
        "n_cells": 6000,
        "simulation_time_before_division": 1500,
        "twin_simulation_time_after_division": twin_time,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": f"{REPO}/simulation_example_input_data/{conn}",
        "param_csv": f"{REPO}/simulation_example_input_data/median_parameter.csv",
        "rows_to_use": [[0, 0]],
        "output_folder": out_dir,
        "log_file": f"{out_dir}/log.jsonl",
        "type": tag,
        "single_k_on": K_ON_MEDIAN,
        "single_K_calib_k_on": k_calib,
        "number_of_cores_per_parameter": 4,
    }


def _exists(out_dir, i, type_name):
    return bool(glob.glob(os.path.join(out_dir, f"df_rows_0_0_{i}_*_{type_name}_*.csv")))


def run_rep(i, cfg, skip):
    if skip and _exists(cfg["output_folder"], i, cfg["type"]):
        print(f"[{cfg['type']} rep {i}] skip", flush=True)
        return None
    set_num_threads(cfg["number_of_cores_per_parameter"])
    t0 = time.time()
    p = process_param_set_single(cfg["rows_to_use"][0], f"rows_0_0_{i}", cfg)
    print(f"[{cfg['type']} rep {i}] {(time.time()-t0)/60:.1f} min -> {p}", flush=True)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, choices=list(NETWORKS))
    ap.add_argument("--k-calib", required=True, type=float, choices=[K_ON_MEDIAN, K_ON_LOW])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--cores-per", type=int, default=4)
    ap.add_argument("--twin-time", type=int, default=300)
    ap.add_argument("--n-cells", type=int, default=6000)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = build_config(args.out_dir, args.network, args.k_calib, args.twin_time)
    cfg["number_of_cores_per_parameter"] = args.cores_per
    cfg["n_cells"] = args.n_cells
    print(f"{args.reps} reps  net={args.network}  k_on=0.66  K-calib@{args.k_calib}  "
          f"type={cfg['type']}  -> {args.out_dir}", flush=True)
    Parallel(n_jobs=args.jobs, backend="multiprocessing")(
        delayed(run_rep)(i, cfg, not args.no_skip_existing) for i in range(1, args.reps + 1))
    print("done", flush=True)
