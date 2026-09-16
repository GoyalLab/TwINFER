"""
Standalone driver for generating the 4 figure_3 networks not already covered
by figure_2 (A_rep_B, A_and_B_both_repress, A_rep_B_B_to_A, A_and_B), 1000
replicates apiece, via twinfer.simulation.gillespie_simulations.process_param_set.

Uses n_cores=1 per replicate throughout: measured on a dedicated node,
n_cores=1/4/20 took 1872.7s/512.8s/132.9s per replicate respectively --
wall-clock per replicate is best with more cores, but *throughput per core*
(replicates/hour for a fixed total core budget) is actually best at
n_cores=1 (parallel efficiency is only ~77% at 4 cores and drops further at
20), so maximizing the number of concurrent single-threaded replicates beats
concentrating cores on fewer replicates -- same lesson as the inference
pipeline.

Output: analysis_data/simulation_data/figure_3_1k/{condition}/ (folder names
match run_causal_direction_v2.py's CONDITIONS dict, so that script can point
--sim-folder at this output directly). One CSV per replicate via
process_param_set's own (timestamp + hash) naming; resumability is done by
globbing for an existing file whose name contains the deterministic
"{condition}_rep_{rep_id}" type string, since process_param_set's output
filename isn't otherwise deterministic.
"""
import argparse
import glob
import os
import time

from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set
from twinfer.utils.paths import get_repo_root

PATH_TO_CODE_REPO = str(get_repo_root())
PATH_TO_INPUT_DATA = f"{PATH_TO_CODE_REPO}/simulation_example_input_data"
PATH_TO_OUTPUT_ROOT = "/home/gzu5140/TwINFER_KA/simulation_data/figure_3_1k"

# condition -> (connectivity matrix stem, rows_to_use row index, output folder)
# Output folder names match run_causal_direction_v2.py's CONDITIONS dict
# values so that script can be pointed at this data directly once it's ready.
NETWORKS = {
    # "A_rep_B": {
    #     "connectivity_matrix": "connectivity_matrix_A_rep_B.txt",
    #     "row": 4,
    #     "output_folder": "A_rep_B",
    # },
    # "A_and_B_both_repress": {
    #     "connectivity_matrix": "connectivity_matrix_A_and_B_both_repress.txt",
    #     "row": 4,
    #     "output_folder": "A_rep_B_B_rep_A",
    # },
    # "A_rep_B_B_to_A": {
    #     "connectivity_matrix": "connectivity_matrix_A_represses_B_B_activates_A.txt",
    #     "row": 5,
    #     "output_folder": "A_rep_B_B_to_A",
    # },
    "A_and_B": {
        "connectivity_matrix": "connectivity_matrix_A_and_B.txt",
        "row": 0,
        "output_folder": "A_to_B_B_to_A",
    },
}

N_REPLICATES = 1000


def build_base_config(condition, n_cores=1):
    spec = NETWORKS[condition]
    output_folder = os.path.join(PATH_TO_OUTPUT_ROOT, spec["output_folder"])
    return {
        "n_cells": 6000,
        "simulation_time_before_division": 1500,
        "twin_simulation_time_after_division": 48,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": f"{PATH_TO_INPUT_DATA}/{spec['connectivity_matrix']}",
        "param_csv": f"{PATH_TO_INPUT_DATA}/median_parameter.csv",
        "rows_to_use": [[spec["row"]] * 2],
        "output_folder": output_folder,
        "log_file": os.path.join(PATH_TO_OUTPUT_ROOT, "logs", f"{condition}.jsonl"),
        "type": None,  # set per replicate
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": n_cores,
    }


def collect_tasks():
    tasks = []
    for condition in NETWORKS:
        for rep_id in range(N_REPLICATES):
            tasks.append((condition, rep_id))
    return tasks


def already_done(output_folder, condition, rep_id):
    pattern = os.path.join(output_folder, f"*{condition}_rep_{rep_id}_*.csv")
    return len(glob.glob(pattern)) > 0


def run_one(condition, rep_id, n_cores=1):
    spec = NETWORKS[condition]
    output_folder = os.path.join(PATH_TO_OUTPUT_ROOT, spec["output_folder"])
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_OUTPUT_ROOT, "logs"), exist_ok=True)

    if already_done(output_folder, condition, rep_id):
        return None, "skipped (already exists)"

    cfg = build_base_config(condition, n_cores=n_cores)
    cfg["type"] = f"{condition}_rep_{rep_id}"

    set_num_threads(n_cores)
    path = process_param_set([spec["row"]] * 2, f"rows_{spec['row']}_{spec['row']}", cfg)
    return path, "computed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=1)
    parser.add_argument("--only", type=str, default=None,
                         help="comma-separated condition:rep_id to run; default = all tasks")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    tasks = collect_tasks()

    if args.list_tasks:
        print("\n".join(f"{c}:{r}" for c, r in tasks))
        raise SystemExit(0)

    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in wanted]

    print(f"Running {len(tasks)} simulation task(s)", flush=True)
    for i, (condition, rep_id) in enumerate(tasks):
        t0 = time.time()
        path, status = run_one(condition, rep_id, n_cores=args.n_cores)
        print(f"[{i+1}/{len(tasks)}] {condition} rep {rep_id}: {status} ({time.time()-t0:.1f}s) -> {path}", flush=True)
